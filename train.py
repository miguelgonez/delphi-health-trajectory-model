import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import List, Callable, Optional, Tuple
import math

from model import DelphiModel, DelphiConfig
from utils import HealthTrajectoryDataset, split_sequences, get_device, set_seed

def estimate_loss(model: DelphiModel, data_loader: DataLoader, device: torch.device, 
                 max_eval_batches: int = 10) -> float:
    """
    Estimate loss on a dataset
    
    Args:
        model: The Delphi model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        max_eval_batches: Maximum number of batches to evaluate
        
    Returns:
        Average loss
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= max_eval_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            logits, loss = model(inputs, targets)
            losses.append(loss.item())
    
    return np.mean(losses)

def get_lr(iteration: int, warmup_iters: int, lr_decay_iters: int, 
           max_lr: float, min_lr: float) -> float:
    """
    Learning rate scheduler with warmup and cosine decay
    
    Args:
        iteration: Current iteration
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of decay iterations
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate for current iteration
    """
    # Linear warmup
    if iteration < warmup_iters:
        return max_lr * iteration / warmup_iters
    
    # Cosine decay
    if iteration > lr_decay_iters:
        return min_lr
    
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train_model(sequences: List[List[int]], config: DelphiConfig, 
                epochs: int = 10, batch_size: int = 32, learning_rate: float = 5e-4,
                weight_decay: float = 1e-1, beta1: float = 0.9, beta2: float = 0.95,
                grad_clip: float = 1.0, warmup_iters: int = 100,
                progress_callback: Optional[Callable] = None) -> Tuple[DelphiModel, List[float]]:
    """
    Train the Delphi model
    
    Args:
        sequences: List of training sequences
        config: Model configuration
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        beta1: Adam optimizer beta1 parameter
        beta2: Adam optimizer beta2 parameter  
        grad_clip: Gradient clipping value
        warmup_iters: Number of warmup iterations
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple of (trained_model, training_losses)
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Training on device: {device}")
    
    # Split data
    train_sequences, val_sequences, _ = split_sequences(sequences, train_ratio=0.8, val_ratio=0.1)
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # Create datasets and data loaders
    train_dataset = HealthTrajectoryDataset(train_sequences, max_length=config.max_seq_len)
    val_dataset = HealthTrajectoryDataset(val_sequences, max_length=config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    # Initialize model
    model = DelphiModel(config)
    model.to(device)
    
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                           weight_decay=weight_decay, betas=(beta1, beta2))
    
    # Training loop
    model.train()
    training_losses = []
    best_val_loss = float('inf')
    iter_num = 0
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Update learning rate
            if iter_num < warmup_iters:
                lr = get_lr(iter_num, warmup_iters, epochs * len(train_loader), 
                           learning_rate, learning_rate * 0.1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Forward pass
            logits, loss = model(inputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Record loss
            epoch_losses.append(loss.item())
            iter_num += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Calculate epoch loss
        epoch_loss = np.mean(epoch_losses)
        training_losses.append(epoch_loss)
        
        # Validation
        val_loss = estimate_loss(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} completed. "
              f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {val_loss:.4f}")
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(epoch, epoch_loss)
        
        # Set model back to training mode
        model.train()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final training loss: {training_losses[-1]:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, training_losses

def save_model(model: DelphiModel, filepath: str):
    """Save trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: str, device: Optional[torch.device] = None) -> DelphiModel:
    """Load trained model"""
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['config']
    
    model = DelphiModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded from {filepath}")
    return model

def fine_tune_model(model: DelphiModel, sequences: List[List[int]], 
                   epochs: int = 5, learning_rate: float = 1e-4,
                   progress_callback: Optional[Callable] = None) -> Tuple[DelphiModel, List[float]]:
    """
    Fine-tune a pre-trained model on new data
    
    Args:
        model: Pre-trained model
        sequences: New training sequences
        epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (fine_tuned_model, fine_tuning_losses)
    """
    device = get_device()
    model.to(device)
    
    # Prepare data
    dataset = HealthTrajectoryDataset(sequences, max_length=model.config.max_seq_len)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # Setup optimizer with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # Fine-tuning loop
    model.train()
    fine_tuning_losses = []
    
    print(f"Starting fine-tuning for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, loss = model(inputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        epoch_loss = np.mean(epoch_losses)
        fine_tuning_losses.append(epoch_loss)
        
        print(f"Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        if progress_callback:
            progress_callback(epoch, epoch_loss)
    
    print("Fine-tuning completed!")
    return model, fine_tuning_losses

def evaluate_perplexity(model: DelphiModel, sequences: List[List[int]]) -> float:
    """
    Evaluate model perplexity on test sequences
    
    Args:
        model: Trained model
        sequences: Test sequences
        
    Returns:
        Perplexity score
    """
    device = get_device()
    model.to(device)
    model.eval()
    
    dataset = HealthTrajectoryDataset(sequences, max_length=model.config.max_seq_len)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, loss = model(inputs, targets)
            
            # Count non-padding tokens
            valid_tokens = (targets != -1).sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self):
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        self.warmup_iters = 100
        self.eval_interval = 100
        self.save_interval = 1000
        self.device = 'auto'  # 'auto', 'cpu', 'cuda', or 'mps'

def create_training_config():
    """Create default training configuration"""
    return TrainingConfig()
