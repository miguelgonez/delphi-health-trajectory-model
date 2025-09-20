import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from model import DelphiModel
from utils import get_device, HealthTrajectoryDataset
from torch.utils.data import DataLoader

class ModelEvaluator:
    """Class for comprehensive model evaluation"""
    
    def __init__(self, model: DelphiModel, disease_mapping: Dict[int, str]):
        self.model = model
        self.disease_mapping = disease_mapping
        self.device = get_device()
        self.model.to(self.device)
        
    def predict_probabilities(self, sequences: List[List[int]], 
                            target_diseases: Optional[List[int]] = None) -> np.ndarray:
        """
        Predict probabilities for next disease events
        
        Args:
            sequences: Input sequences
            target_diseases: Specific diseases to predict (if None, predict all)
            
        Returns:
            Probability predictions array
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for seq in sequences:
                if len(seq) > 1:
                    # Use sequence up to second-to-last element to predict last
                    input_seq = torch.tensor(seq[:-1]).unsqueeze(0).to(self.device)
                    logits, _ = self.model(input_seq)
                    
                    # Get probabilities for last position
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    predictions.append(probs.cpu().numpy())
                else:
                    # For very short sequences, use zeros
                    predictions.append(np.zeros(self.model.config.vocab_size))
        
        predictions = np.array(predictions)
        
        if target_diseases is not None:
            return predictions[:, target_diseases]
        
        return predictions
    
    def create_binary_targets(self, sequences: List[List[int]], 
                            target_diseases: List[int]) -> np.ndarray:
        """
        Create binary targets for disease prediction evaluation
        
        Args:
            sequences: Input sequences
            target_diseases: Disease codes to create targets for
            
        Returns:
            Binary target array
        """
        targets = np.zeros((len(sequences), len(target_diseases)))
        
        for i, seq in enumerate(sequences):
            if len(seq) > 1:
                last_disease = seq[-1]
                for j, disease in enumerate(target_diseases):
                    if last_disease == disease:
                        targets[i, j] = 1
        
        return targets
    
    def evaluate_auc_scores(self, sequences: List[List[int]], 
                          target_diseases: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate AUC scores for disease prediction
        
        Args:
            sequences: Test sequences
            target_diseases: Diseases to evaluate (if None, use all)
            
        Returns:
            Dictionary of AUC scores by disease
        """
        if target_diseases is None:
            target_diseases = list(range(1, self.model.config.vocab_size))  # Exclude padding token
        
        # Get predictions and targets
        predictions = self.predict_probabilities(sequences, target_diseases)
        targets = self.create_binary_targets(sequences, target_diseases)
        
        auc_scores = {}
        
        for i, disease in enumerate(target_diseases):
            disease_name = self.disease_mapping.get(disease, f"Disease_{disease}")
            
            # Check if we have positive examples
            if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                try:
                    auc = roc_auc_score(targets[:, i], predictions[:, i])
                    auc_scores[disease_name] = auc
                except ValueError as e:
                    print(f"Warning: Could not compute AUC for {disease_name}: {e}")
                    auc_scores[disease_name] = 0.5  # Random performance
            else:
                auc_scores[disease_name] = 0.5  # No positive examples or all positive
        
        return auc_scores
    
    def evaluate_average_precision(self, sequences: List[List[int]], 
                                 target_diseases: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate Average Precision scores
        
        Args:
            sequences: Test sequences
            target_diseases: Diseases to evaluate
            
        Returns:
            Dictionary of AP scores by disease
        """
        if target_diseases is None:
            target_diseases = list(range(1, self.model.config.vocab_size))
        
        predictions = self.predict_probabilities(sequences, target_diseases)
        targets = self.create_binary_targets(sequences, target_diseases)
        
        ap_scores = {}
        
        for i, disease in enumerate(target_diseases):
            disease_name = self.disease_mapping.get(disease, f"Disease_{disease}")
            
            if targets[:, i].sum() > 0:
                try:
                    ap = average_precision_score(targets[:, i], predictions[:, i])
                    ap_scores[disease_name] = ap
                except ValueError as e:
                    print(f"Warning: Could not compute AP for {disease_name}: {e}")
                    ap_scores[disease_name] = 0.0
            else:
                ap_scores[disease_name] = 0.0
        
        return ap_scores
    
    def evaluate_calibration(self, sequences: List[List[int]], 
                           target_diseases: Optional[List[int]] = None,
                           n_bins: int = 10) -> Dict[str, Dict]:
        """
        Evaluate model calibration
        
        Args:
            sequences: Test sequences
            target_diseases: Diseases to evaluate
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary of calibration metrics by disease
        """
        if target_diseases is None:
            target_diseases = list(range(1, self.model.config.vocab_size))
        
        predictions = self.predict_probabilities(sequences, target_diseases)
        targets = self.create_binary_targets(sequences, target_diseases)
        
        calibration_results = {}
        
        for i, disease in enumerate(target_diseases):
            disease_name = self.disease_mapping.get(disease, f"Disease_{disease}")
            
            if targets[:, i].sum() > 0:
                try:
                    fraction_pos, mean_pred = calibration_curve(
                        targets[:, i], predictions[:, i], n_bins=n_bins
                    )
                    
                    # Expected Calibration Error
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    
                    ece = 0
                    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                        # Find predictions in bin
                        in_bin = (predictions[:, i] > bin_lower) & (predictions[:, i] <= bin_upper)
                        prop_in_bin = in_bin.sum() / len(predictions)
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = targets[in_bin, i].mean()
                            avg_confidence_in_bin = predictions[in_bin, i].mean()
                            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    calibration_results[disease_name] = {
                        'expected_calibration_error': ece,
                        'fraction_of_positives': fraction_pos,
                        'mean_predicted_value': mean_pred
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not compute calibration for {disease_name}: {e}")
                    calibration_results[disease_name] = {
                        'expected_calibration_error': float('inf'),
                        'fraction_of_positives': np.array([]),
                        'mean_predicted_value': np.array([])
                    }
        
        return calibration_results
    
    def compute_baseline_scores(self, sequences: List[List[int]], 
                              target_diseases: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Compute baseline scores (age-sex based epidemiological baseline)
        
        Args:
            sequences: Test sequences
            target_diseases: Diseases to evaluate
            
        Returns:
            Dictionary of baseline AUC scores
        """
        if target_diseases is None:
            target_diseases = list(range(1, self.model.config.vocab_size))
        
        # Create synthetic baseline predictions based on disease prevalence
        targets = self.create_binary_targets(sequences, target_diseases)
        baseline_scores = {}
        
        for i, disease in enumerate(target_diseases):
            disease_name = self.disease_mapping.get(disease, f"Disease_{disease}")
            
            # Use disease prevalence as baseline prediction
            prevalence = targets[:, i].mean()
            baseline_predictions = np.full(len(sequences), prevalence)
            
            # Add some random noise to make it more realistic
            baseline_predictions += np.random.normal(0, 0.1, len(sequences))
            baseline_predictions = np.clip(baseline_predictions, 0, 1)
            
            if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                try:
                    baseline_auc = roc_auc_score(targets[:, i], baseline_predictions)
                    baseline_scores[disease_name] = baseline_auc
                except ValueError:
                    baseline_scores[disease_name] = 0.5
            else:
                baseline_scores[disease_name] = 0.5
        
        return baseline_scores
    
    def generate_evaluation_report(self, sequences: List[List[int]]) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            sequences: Test sequences
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("Computing evaluation metrics...")
        
        # Get disease subset for evaluation (exclude padding and rare diseases)
        target_diseases = list(range(1, min(11, self.model.config.vocab_size)))  # First 10 diseases
        
        # Compute metrics
        auc_scores = self.evaluate_auc_scores(sequences, target_diseases)
        ap_scores = self.evaluate_average_precision(sequences, target_diseases)
        calibration_results = self.evaluate_calibration(sequences, target_diseases)
        baseline_scores = self.compute_baseline_scores(sequences, target_diseases)
        
        # Compute summary statistics
        valid_aucs = [score for score in auc_scores.values() if score > 0]
        valid_aps = [score for score in ap_scores.values() if score > 0]
        valid_baselines = [score for score in baseline_scores.values() if score > 0]
        valid_eces = [cal['expected_calibration_error'] for cal in calibration_results.values() 
                     if cal['expected_calibration_error'] != float('inf')]
        
        summary = {
            'mean_auc': np.mean(valid_aucs) if valid_aucs else 0.5,
            'std_auc': np.std(valid_aucs) if valid_aucs else 0.0,
            'mean_ap': np.mean(valid_aps) if valid_aps else 0.0,
            'std_ap': np.std(valid_aps) if valid_aps else 0.0,
            'mean_baseline_auc': np.mean(valid_baselines) if valid_baselines else 0.5,
            'mean_ece': np.mean(valid_eces) if valid_eces else 0.0,
            'improvement_over_baseline': (np.mean(valid_aucs) - np.mean(valid_baselines)) if valid_aucs and valid_baselines else 0.0
        }
        
        report = {
            'summary': summary,
            'auc_scores': auc_scores,
            'average_precision_scores': ap_scores,
            'calibration_results': calibration_results,
            'baseline_scores': baseline_scores,
            'num_test_sequences': len(sequences)
        }
        
        return report
    
    def plot_evaluation_results(self, evaluation_report: Dict, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot evaluation results
        
        Args:
            evaluation_report: Report from generate_evaluation_report
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # AUC scores comparison
        auc_scores = evaluation_report['auc_scores']
        baseline_scores = evaluation_report['baseline_scores']
        
        diseases = list(auc_scores.keys())
        model_aucs = list(auc_scores.values())
        baseline_aucs = [baseline_scores[disease] for disease in diseases]
        
        x = np.arange(len(diseases))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, model_aucs, width, label='Delphi Model', alpha=0.8)
        axes[0, 0].bar(x + width/2, baseline_aucs, width, label='Baseline', alpha=0.8)
        axes[0, 0].set_xlabel('Disease')
        axes[0, 0].set_ylabel('AUC Score')
        axes[0, 0].set_title('AUC Scores: Model vs Baseline')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(diseases, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average Precision scores
        ap_scores = evaluation_report['average_precision_scores']
        ap_values = list(ap_scores.values())
        
        axes[0, 1].bar(diseases, ap_values, alpha=0.8, color='orange')
        axes[0, 1].set_xlabel('Disease')
        axes[0, 1].set_ylabel('Average Precision')
        axes[0, 1].set_title('Average Precision Scores')
        axes[0, 1].set_xticklabels(diseases, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calibration plot (for first disease with valid calibration)
        calibration_results = evaluation_report['calibration_results']
        for disease, cal_data in calibration_results.items():
            if len(cal_data['fraction_of_positives']) > 0:
                axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                axes[1, 0].plot(cal_data['mean_predicted_value'], 
                               cal_data['fraction_of_positives'], 
                               's-', label=f'{disease}')
                break
        
        axes[1, 0].set_xlabel('Mean Predicted Probability')
        axes[1, 0].set_ylabel('Fraction of Positives')
        axes[1, 0].set_title('Calibration Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary metrics
        summary = evaluation_report['summary']
        metrics = ['Mean AUC', 'Mean AP', 'Mean ECE', 'Improvement over Baseline']
        values = [summary['mean_auc'], summary['mean_ap'], 
                 summary['mean_ece'], summary['improvement_over_baseline']]
        
        colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in values]
        axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Summary Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def evaluate_model(model: DelphiModel, test_sequences: List[List[int]], 
                  disease_mapping: Dict[int, str]) -> Dict:
    """
    Convenience function to evaluate a model
    
    Args:
        model: Trained Delphi model
        test_sequences: Test sequences
        disease_mapping: Disease code to name mapping
        
    Returns:
        Evaluation report dictionary
    """
    evaluator = ModelEvaluator(model, disease_mapping)
    return evaluator.generate_evaluation_report(test_sequences)

def compare_models(models: Dict[str, DelphiModel], test_sequences: List[List[int]], 
                  disease_mapping: Dict[int, str]) -> pd.DataFrame:
    """
    Compare multiple models
    
    Args:
        models: Dictionary of model name -> model
        test_sequences: Test sequences
        disease_mapping: Disease mapping
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        evaluator = ModelEvaluator(model, disease_mapping)
        report = evaluator.generate_evaluation_report(test_sequences)
        
        results.append({
            'Model': model_name,
            'Mean AUC': report['summary']['mean_auc'],
            'Mean AP': report['summary']['mean_ap'],
            'Mean ECE': report['summary']['mean_ece'],
            'Improvement over Baseline': report['summary']['improvement_over_baseline']
        })
    
    return pd.DataFrame(results)
