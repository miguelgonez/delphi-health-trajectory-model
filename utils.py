import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
import random
import os

class DiseaseTokenizer:
    """Centralized tokenizer for diseases with PAD=0 and contiguous token IDs"""
    
    def __init__(self):
        self.name_to_token = {}
        self.token_to_name = {}
        self.vocab_size = 1  # Start with 1 for PAD token
        
        # PAD token is always 0
        self.token_to_name[0] = "PAD"
        
        # Load disease mappings
        try:
            labels_df = pd.read_csv('delphi_labels_chapters_colours_icd.csv')
            diseases = labels_df['disease_name'].tolist()
        except:
            # Fallback diseases
            diseases = [
                'Hypertension', 'Diabetes', 'Coronary Artery Disease', 'Stroke', 'Cancer',
                'Depression', 'Anxiety', 'Asthma', 'COPD', 'Arthritis', 'Osteoporosis',
                'Kidney Disease', 'Liver Disease', 'Heart Failure', 'Atrial Fibrillation'
            ]
        
        # Assign contiguous token IDs starting from 1
        for i, disease in enumerate(diseases):
            token_id = i + 1
            self.name_to_token[disease] = token_id
            self.token_to_name[token_id] = disease
        
        self.vocab_size = len(diseases) + 1  # +1 for PAD token
    
    def encode(self, disease_name: str) -> int:
        """Convert disease name to token ID"""
        return self.name_to_token.get(disease_name, 0)  # Default to PAD if unknown
    
    def decode(self, token_id: int) -> str:
        """Convert token ID to disease name"""
        return self.token_to_name.get(token_id, f"Unknown_{token_id}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size including PAD token"""
        return self.vocab_size
    
    def get_disease_names(self) -> List[str]:
        """Get list of all disease names (excluding PAD)"""
        return [name for token_id, name in self.token_to_name.items() if token_id != 0]

# Global tokenizer instance
_tokenizer = None

def get_tokenizer() -> DiseaseTokenizer:
    """Get global disease tokenizer instance"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = DiseaseTokenizer()
    return _tokenizer

def get_disease_mapping() -> Dict[str, int]:
    """Get unified disease name to token ID mapping"""
    tokenizer = get_tokenizer()
    return tokenizer.name_to_token

def get_code_to_name_mapping() -> Dict[int, str]:
    """Get unified token ID to disease name mapping"""
    tokenizer = get_tokenizer()
    return tokenizer.token_to_name

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def prepare_data(raw_data: pd.DataFrame) -> Tuple[List[List[int]], List[List[float]], List[List[str]]]:
    """
    Prepare raw health trajectory data for model training
    
    Args:
        raw_data: DataFrame with patient health trajectories
        
    Returns:
        Tuple of (sequences, ages, dates) for training and visualization
    """
    sequences = []
    ages_per_patient = []
    dates_per_patient = []
    
    if not raw_data.empty and 'patient_id' in raw_data.columns:
        # Process real data from CSV
        for patient_id in raw_data['patient_id'].unique():
            patient_data = raw_data[raw_data['patient_id'] == patient_id].copy()
            
            # Sort by age or date if available
            if 'age' in patient_data.columns:
                patient_data = patient_data.sort_values('age')
            elif 'event_date' in patient_data.columns:
                patient_data['event_date'] = pd.to_datetime(patient_data['event_date'])
                patient_data = patient_data.sort_values('event_date')
            
            # Get disease codes and normalize through tokenizer
            tokenizer = get_tokenizer()
            if 'disease_name' in patient_data.columns:
                # Preferred: encode disease names through tokenizer
                disease_codes = [tokenizer.encode(name) for name in patient_data['disease_name'].tolist()]
            elif 'disease_code' in patient_data.columns:
                # If disease_code is available, assume it matches tokenizer IDs, but validate
                disease_codes = patient_data['disease_code'].tolist()
                # Validate that all codes are within tokenizer range
                valid_codes = []
                for code in disease_codes:
                    if 0 <= code < tokenizer.get_vocab_size():
                        valid_codes.append(code)
                    else:
                        valid_codes.append(0)  # Replace invalid codes with PAD
                disease_codes = valid_codes
            else:
                # Fallback: no disease data
                disease_codes = [0]  # PAD token
            
            # Get ages
            if 'age' in patient_data.columns:
                ages = patient_data['age'].tolist()
            else:
                # Generate ages based on typical progression
                base_age = np.random.uniform(25, 65)
                ages = [base_age + i * np.random.uniform(0.5, 3) for i in range(len(disease_codes))]
            
            # Get dates
            if 'event_date' in patient_data.columns:
                dates = patient_data['event_date'].dt.strftime('%Y-%m-%d').tolist()
            else:
                # Generate realistic dates
                base_year = 2020
                dates = [f"{base_year + int(i/2)}-{(i%12)+1:02d}-{np.random.randint(1,28):02d}" 
                        for i in range(len(disease_codes))]
            
            if disease_codes:  # Only add if we have data
                sequences.append(disease_codes)
                ages_per_patient.append(ages)
                dates_per_patient.append(dates)
                
            if len(sequences) >= 1000:  # Limit for performance
                break
    
    # Only use real data - no synthetic padding
    
    return sequences, ages_per_patient, dates_per_patient

def encode_sequences(sequences: List[List[int]], max_length: int = 256) -> torch.Tensor:
    """
    Encode sequences as tensors for training
    
    Args:
        sequences: List of disease sequences
        max_length: Maximum sequence length for padding/truncation
        
    Returns:
        Tensor of encoded sequences
    """
    encoded = []
    
    for seq in sequences:
        # Truncate if too long
        if len(seq) > max_length:
            seq = seq[:max_length]
        
        # Pad if too short
        if len(seq) < max_length:
            seq = seq + [0] * (max_length - len(seq))  # 0 is padding token
        
        encoded.append(seq)
    
    return torch.tensor(encoded, dtype=torch.long)

def decode_sequences(encoded_sequences: torch.Tensor, disease_mapping: Optional[Dict[int, str]] = None) -> List[List[str]]:
    """
    Decode tensor sequences back to disease names
    
    Args:
        encoded_sequences: Tensor of encoded sequences
        disease_mapping: Optional mapping from codes to disease names
        
    Returns:
        List of decoded sequences
    """
    if disease_mapping is None:
        # Use centralized tokenizer mapping
        tokenizer = get_tokenizer()
        disease_mapping = tokenizer.token_to_name
    
    decoded = []
    for seq in encoded_sequences:
        seq_diseases = []
        for code in seq:
            code_int = code.item() if torch.is_tensor(code) else code
            if code_int in disease_mapping and code_int != 0:  # Skip padding
                seq_diseases.append(disease_mapping[code_int])
        decoded.append(seq_diseases)
    
    return decoded

def create_batches(sequences: List[List[int]], batch_size: int = 32) -> List[torch.Tensor]:
    """
    Create batches for training
    
    Args:
        sequences: List of sequences
        batch_size: Size of each batch
        
    Returns:
        List of batched tensors
    """
    batches = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        
        # Find max length in batch
        max_len = max(len(seq) for seq in batch_sequences)
        
        # Pad sequences to same length
        padded_batch = []
        for seq in batch_sequences:
            if len(seq) < max_len:
                padded_seq = seq + [0] * (max_len - len(seq))
            else:
                padded_seq = seq
            padded_batch.append(padded_seq)
        
        batches.append(torch.tensor(padded_batch, dtype=torch.long))
    
    return batches

def compute_sequence_statistics(sequences: List[List[int]]) -> Dict:
    """
    Compute statistics about the sequences
    
    Args:
        sequences: List of sequences
        
    Returns:
        Dictionary with statistics
    """
    lengths = [len(seq) for seq in sequences]
    all_diseases = [disease for seq in sequences for disease in seq if disease != 0]
    
    stats = {
        'num_sequences': len(sequences),
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'total_events': len(all_diseases),
        'unique_diseases': len(set(all_diseases)),
        'disease_frequencies': {disease: all_diseases.count(disease) for disease in set(all_diseases)}
    }
    
    return stats

def split_sequences(sequences: List[List[int]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List, List, List]:
    """
    Split sequences into train, validation, and test sets
    
    Args:
        sequences: List of sequences
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        Tuple of (train, validation, test) sequences
    """
    n_total = len(sequences)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Shuffle sequences
    shuffled_sequences = sequences.copy()
    random.shuffle(shuffled_sequences)
    
    train_sequences = shuffled_sequences[:n_train]
    val_sequences = shuffled_sequences[n_train:n_train + n_val]
    test_sequences = shuffled_sequences[n_train + n_val:]
    
    return train_sequences, val_sequences, test_sequences

def create_prediction_targets(sequences: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Create input-target pairs for next-token prediction
    
    Args:
        sequences: List of sequences
        
    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    input_sequences = []
    target_sequences = []
    
    for seq in sequences:
        if len(seq) > 1:
            # Input: all tokens except the last
            # Target: all tokens except the first (shifted by 1)
            input_sequences.append(seq[:-1])
            target_sequences.append(seq[1:])
    
    return input_sequences, target_sequences

def save_sequences(sequences: List[List[int]], filepath: str):
    """Save sequences to file"""
    np.save(filepath, sequences)

def load_sequences(filepath: str) -> List[List[int]]:
    """Load sequences from file"""
    return np.load(filepath, allow_pickle=True).tolist()

def validate_sequences(sequences: List[List[int]], vocab_size: int = 16) -> bool:
    """
    Validate that sequences contain only valid tokens
    
    Args:
        sequences: List of sequences to validate
        vocab_size: Size of vocabulary
        
    Returns:
        True if all sequences are valid
    """
    for seq in sequences:
        for token in seq:
            if not (0 <= token < vocab_size):
                return False
    return True

def augment_sequences(sequences: List[List[int]], augmentation_factor: float = 0.1) -> List[List[int]]:
    """
    Augment sequences by adding noise (for data augmentation)
    
    Args:
        sequences: Original sequences
        augmentation_factor: Fraction of sequences to augment
        
    Returns:
        Augmented sequences
    """
    augmented = sequences.copy()
    n_augment = int(len(sequences) * augmentation_factor)
    
    for _ in range(n_augment):
        # Choose random sequence to augment
        orig_seq = random.choice(sequences)
        
        if len(orig_seq) > 2:
            # Create augmentation by randomly dropping one element
            aug_seq = orig_seq.copy()
            drop_idx = random.randint(1, len(aug_seq) - 2)  # Don't drop first or last
            aug_seq.pop(drop_idx)
            augmented.append(aug_seq)
    
    return augmented

class HealthTrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for health trajectories"""
    
    def __init__(self, sequences: List[List[int]], max_length: int = 256):
        self.sequences = sequences
        self.max_length = max_length
        
        # Create input-target pairs
        self.inputs, self.targets = create_prediction_targets(sequences)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        target_seq = self.targets[idx]
        
        # Pad sequences
        if len(input_seq) < self.max_length:
            input_seq = input_seq + [0] * (self.max_length - len(input_seq))
            target_seq = target_seq + [-1] * (self.max_length - len(target_seq))  # -1 for ignored tokens
        else:
            input_seq = input_seq[:self.max_length]
            target_seq = target_seq[:self.max_length]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
