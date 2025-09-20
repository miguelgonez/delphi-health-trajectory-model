import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set style
plt.style.use('default')
sns.set_palette("husl")

def plot_trajectory(patient_data: Dict, disease_mapping: Dict[int, str], 
                   patient_id: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
    """
    Plot a patient's health trajectory timeline
    
    Args:
        patient_data: Dictionary with 'ages', 'diseases', and 'events'
        disease_mapping: Mapping from disease codes to names
        patient_id: Optional patient identifier
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Extract data
    ages = patient_data.get('ages', [])
    diseases = patient_data.get('diseases', [])
    events = patient_data.get('events', diseases)  # Use diseases if events not provided
    
    # Create timeline plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(diseases))))
    disease_colors = {disease: colors[i] for i, disease in enumerate(set(diseases))}
    
    # Plot events as points on timeline
    for i, (age, disease) in enumerate(zip(ages, diseases)):
        disease_name = disease_mapping.get(disease, f"Disease_{disease}")
        color = disease_colors[disease]
        
        ax1.scatter(age, i, c=[color], s=100, alpha=0.8)
        ax1.text(age + 0.5, i, disease_name, fontsize=9, va='center')
    
    ax1.set_xlabel('Age (years)')
    ax1.set_ylabel('Event Sequence')
    title = f'Health Trajectory for Patient {patient_id}' if patient_id else 'Health Trajectory'
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    # Plot age distribution of events
    ax2.hist(ages, bins=min(20, len(ages)), alpha=0.7, color='lightblue', edgecolor='black')
    ax2.set_xlabel('Age (years)')
    ax2.set_ylabel('Event Count')
    ax2.set_title('Distribution of Event Ages')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_disease_cooccurrence(sequences: List[List[int]], disease_mapping: Dict[int, str],
                            figsize: Tuple[int, int] = (10, 8)):
    """
    Plot disease co-occurrence matrix
    
    Args:
        sequences: List of disease sequences
        disease_mapping: Mapping from disease codes to names
        figsize: Figure size
    """
    # Get unique diseases
    all_diseases = set()
    for seq in sequences:
        all_diseases.update(seq)
    all_diseases.discard(0)  # Remove padding token
    all_diseases = sorted(list(all_diseases))
    
    # Create co-occurrence matrix
    cooccurrence = np.zeros((len(all_diseases), len(all_diseases)))
    
    for seq in sequences:
        seq_diseases = [d for d in seq if d != 0]  # Remove padding
        for i, disease1 in enumerate(seq_diseases):
            for disease2 in seq_diseases[i+1:]:  # Only count each pair once
                if disease1 in all_diseases and disease2 in all_diseases:
                    idx1 = all_diseases.index(disease1)
                    idx2 = all_diseases.index(disease2)
                    cooccurrence[idx1, idx2] += 1
                    cooccurrence[idx2, idx1] += 1  # Make symmetric
    
    # Create labels
    labels = [disease_mapping.get(d, f"Disease_{d}") for d in all_diseases]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cooccurrence, annot=True, fmt='g', xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', ax=ax)
    ax.set_title('Disease Co-occurrence Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def plot_attention(attention_weights: np.ndarray, tokens: List[str], 
                  layer_idx: int = 0, head_idx: int = 0, figsize: Tuple[int, int] = (10, 8)):
    """
    Plot attention weights heatmap
    
    Args:
        attention_weights: Attention weights array [batch, heads, seq_len, seq_len]
        tokens: List of token names
        layer_idx: Layer index
        head_idx: Attention head index
        figsize: Figure size
    """
    # Extract attention for specific layer and head
    if attention_weights.ndim == 4:
        att_matrix = attention_weights[0, head_idx, :len(tokens), :len(tokens)]
    else:
        att_matrix = attention_weights[:len(tokens), :len(tokens)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(att_matrix, annot=True, fmt='.2f', xticklabels=tokens, yticklabels=tokens,
                cmap='Blues', ax=ax)
    
    ax.set_title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def plot_umap_embeddings(embeddings: np.ndarray, labels: List[str], 
                        title: str = "UMAP Projection of Disease Embeddings",
                        figsize: Tuple[int, int] = (12, 8)):
    """
    Plot UMAP projection of embeddings
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Labels for each embedding
        title: Plot title
        figsize: Figure size
    """
    # Apply UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'UMAP1': embedding_2d[:, 0],
        'UMAP2': embedding_2d[:, 1],
        'Disease': labels
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    scatter = ax.scatter(df['UMAP1'], df['UMAP2'], c=range(len(labels)), 
                        cmap='tab20', s=100, alpha=0.7)
    
    # Add labels
    for i, (x, y, label) in enumerate(zip(df['UMAP1'], df['UMAP2'], df['Disease'])):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_training_loss(losses: List[float], title: str = "Training Loss", 
                      figsize: Tuple[int, int] = (10, 6)):
    """
    Plot training loss curve
    
    Args:
        losses: List of loss values
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(losses) > 1:
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_disease_progression(sequence: List[int], ages: List[float], 
                           disease_mapping: Dict[int, str],
                           figsize: Tuple[int, int] = (12, 6)):
    """
    Plot disease progression over time
    
    Args:
        sequence: Sequence of disease codes
        ages: Corresponding ages for each disease
        disease_mapping: Mapping from codes to names
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out padding tokens
    valid_indices = [i for i, disease in enumerate(sequence) if disease != 0]
    valid_diseases = [sequence[i] for i in valid_indices]
    valid_ages = [ages[i] for i in valid_indices]
    
    # Create timeline
    y_positions = range(len(valid_diseases))
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(valid_diseases))))
    disease_colors = {disease: colors[i] for i, disease in enumerate(set(valid_diseases))}
    
    for i, (age, disease) in enumerate(zip(valid_ages, valid_diseases)):
        disease_name = disease_mapping.get(disease, f"Disease_{disease}")
        color = disease_colors[disease]
        
        # Plot point
        ax.scatter(age, i, c=[color], s=150, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add disease name
        ax.text(age + 0.5, i, disease_name, fontsize=10, va='center', ha='left')
        
        # Add connecting line
        if i > 0:
            ax.plot([valid_ages[i-1], age], [i-1, i], 'gray', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Disease Event Sequence')
    ax.set_title('Disease Progression Timeline')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to show all events
    ax.set_ylim(-0.5, len(valid_diseases) - 0.5)
    
    plt.tight_layout()
    return fig

def plot_risk_predictions(risk_scores: np.ndarray, disease_mapping: Dict[int, str],
                         time_horizons: List[str], figsize: Tuple[int, int] = (12, 8)):
    """
    Plot risk predictions over different time horizons
    
    Args:
        risk_scores: Array of risk scores [time_steps, diseases]
        disease_mapping: Mapping from codes to names
        time_horizons: List of time horizon labels
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top diseases by average risk
    avg_risks = np.mean(risk_scores, axis=0)
    top_disease_indices = np.argsort(avg_risks)[-10:]  # Top 10 diseases
    
    # Plot risk evolution
    for disease_idx in top_disease_indices:
        disease_name = disease_mapping.get(disease_idx, f"Disease_{disease_idx}")
        ax.plot(time_horizons, risk_scores[:, disease_idx], 
               marker='o', linewidth=2, label=disease_name)
    
    ax.set_xlabel('Time Horizon')
    ax.set_ylabel('Risk Score')
    ax.set_title('Disease Risk Predictions Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                          n_bins: int = 10, figsize: Tuple[int, int] = (8, 6)):
    """
    Plot calibration curve
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        figsize: Figure size
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
           label='Model Calibration', linewidth=2, markersize=8)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_roc_curves(y_true_dict: Dict[str, np.ndarray], y_prob_dict: Dict[str, np.ndarray],
                   figsize: Tuple[int, int] = (10, 8)):
    """
    Plot ROC curves for multiple diseases
    
    Args:
        y_true_dict: Dictionary of true labels for each disease
        y_prob_dict: Dictionary of predicted probabilities for each disease
        figsize: Figure size
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(y_true_dict)))
    
    for i, (disease, y_true) in enumerate(y_true_dict.items()):
        y_prob = y_prob_dict[disease]
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
               label=f'{disease} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves by Disease')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_interactive_trajectory_plot(patient_data: Dict, disease_mapping: Dict[int, str],
                                     patient_id: Optional[str] = None):
    """
    Create interactive trajectory plot using Plotly
    
    Args:
        patient_data: Dictionary with patient trajectory data
        disease_mapping: Mapping from disease codes to names
        patient_id: Optional patient identifier
    """
    ages = patient_data.get('ages', [])
    diseases = patient_data.get('diseases', [])
    
    # Create disease names
    disease_names = [disease_mapping.get(d, f"Disease_{d}") for d in diseases]
    
    # Create the plot
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=ages,
        y=list(range(len(ages))),
        mode='markers+text+lines',
        text=disease_names,
        textposition="middle right",
        marker=dict(size=12, color=diseases, colorscale='Set3'),
        line=dict(color='gray', width=1),
        name='Health Trajectory'
    ))
    
    title = f'Interactive Health Trajectory for Patient {patient_id}' if patient_id else 'Interactive Health Trajectory'
    fig.update_layout(
        title=title,
        xaxis_title='Age (years)',
        yaxis_title='Event Sequence',
        hovermode='closest',
        height=600
    )
    
    return fig
