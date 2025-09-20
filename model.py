import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DelphiConfig:
    """Configuration class for Delphi model"""
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    max_seq_len: int = 256
    vocab_size: int = 16  # Number of disease types + special tokens

class LayerNorm(nn.Module):
    """LayerNorm with optional bias"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Causal self-attention module"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Flash attention flag
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask
            self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                               .view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """Multi-layer perceptron"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DelphiModel(nn.Module):
    """Delphi transformer model for health trajectory modeling"""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_seq_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.max_seq_len, config.n_embd), # Position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight sharing between token embeddings and output layer
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Cannot forward sequence of length {t}, max seq length is {self.config.max_seq_len}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model, but want to use a smaller block size for some task
        assert block_size <= self.config.max_seq_len
        self.config.max_seq_len = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def get_attention_weights(self, idx, layer_idx=0):
        """Extract attention weights for visualization"""
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Forward through layers until target layer
        for i, block in enumerate(self.transformer.h):
            if i == layer_idx:
                # Get attention weights from this layer
                x_norm = block.ln_1(x)
                q, k, v = block.attn.c_attn(x_norm).split(self.config.n_embd, dim=2)
                k = k.view(b, t, self.config.n_head, self.config.n_embd // self.config.n_head).transpose(1, 2)
                q = q.view(b, t, self.config.n_head, self.config.n_embd // self.config.n_head).transpose(1, 2)
                
                # Compute attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = F.softmax(att, dim=-1)
                
                return att.detach().cpu().numpy()
            
            x = block(x)
        
        return None

    def get_embeddings(self, idx):
        """Get token embeddings for UMAP visualization"""
        return self.transformer.wte(idx).detach().cpu().numpy()

    def predict_next_disease(self, sequence, temperature=1.0):
        """Predict probability distribution over next diseases"""
        self.eval()
        with torch.no_grad():
            idx = torch.tensor(sequence).unsqueeze(0)  # Add batch dimension
            logits, _ = self(idx)
            
            # Get last position logits and apply temperature
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            return probs.cpu().numpy()

    def compute_risk_scores(self, sequence, horizon_steps=5):
        """Compute risk scores for future disease events"""
        self.eval()
        with torch.no_grad():
            idx = torch.tensor(sequence).unsqueeze(0)
            
            # Generate future sequence
            future_sequence = self.generate(idx, horizon_steps, temperature=0.8)
            
            # Compute probabilities for each future step
            risk_scores = []
            for i in range(1, horizon_steps + 1):
                current_seq = future_sequence[:, :-horizon_steps+i] if i < horizon_steps else future_sequence
                logits, _ = self(current_seq)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                risk_scores.append(probs.cpu().numpy())
            
            return np.array(risk_scores)

def create_delphi_model(config=None):
    """Factory function to create a Delphi model"""
    if config is None:
        config = DelphiConfig()
    
    model = DelphiModel(config)
    return model
