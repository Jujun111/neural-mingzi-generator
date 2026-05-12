import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

class ChineseNameLSTM(nn.Module):
    """
    Character-level LSTM model for generating Chinese names.
    Provides semantic embeddings for characters instead of frequentist transition probabilities.
    """
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int = 128, 
                 hidden_size: int = 256, 
                 num_layers: int = 1, 
                 pad_idx: int = 0,
                 dropout_prob: float = 0.3):
        super(ChineseNameLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )

        # 2. LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # Framework dropout only applies if num_layers > 1
            dropout=dropout_prob if num_layers > 1 else 0.0 
        )

        # 3. Regularization Layer
        # Applied to the output of the LSTM to prevent memorization
        self.dropout = nn.Dropout(p=dropout_prob)

        # 4. Fully Connected (Output) Layer
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
        )

    def forward(self, 
                x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for processing batches during training.
        
        Args:
            x (Tensor): Input sequences. Shape: (batch_size, seq_len).
            hidden (tuple, optional): (h_0, c_0). PyTorch auto-initializes to zeros if None.
            
        Returns:
            logits (Tensor): Unnormalized predictions. Shape: (batch_size, seq_len, vocab_size).
            hidden (tuple): Updated hidden cell states (h_n, c_n).
        """
        # 1. Embed inputs -> Shape: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # 2. Pass through LSTM -> Output Shape: (batch_size, seq_len, hidden_size)
        # We pass hidden directly. If None, PyTorch handles device-aware zero initialization in C++
        output, hidden = self.lstm(embedded, hidden)
        
        # 3. Apply Dropout regularization
        output = self.dropout(output)
        
        # 4. Pass through Classifier Layer -> Output Shape: (batch_size, seq_len, vocab_size)
        logits = self.fc(output)
        
        return logits, hidden

class ChineseNameTransformer(nn.Module):
    """
    Autoregressive Decoder-Only Transformer for Sequence Generation.
    """
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int = 256, 
                 num_heads: int = 8, 
                 hidden_dim: int = 512, 
                 num_layers: int = 2, 
                 max_seq_len: int = 10, 
                 pad_idx: int = 0,
                 dropout_prob: float = 0.3):
        super(ChineseNameTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx
        
        # 1. Token Embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=pad_idx
        )
        
        # 2. Positional Embeddings
        # Unlike LSTMs, Transformers have no concept of sequence order.
        # We learn an embedding vector for each absolute position (0 to max_seq_len).
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_seq_len, 
            embedding_dim=embedding_dim
        )
        
        # 3. Transformer Encoder Blocks (Used causally via masking)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 4. Regularization & Projection
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc_out = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on the diagonal.
        This prevents the self-attention mechanism from attending to future tokens.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x (Tensor): Input sequences. Shape: (batch_size, seq_len).
            hidden: Ignored, kept for compatibility with LSTM signature.
            
        Returns:
            logits (Tensor): Unnormalized predictions. Shape: (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = x.size()
        device = x.device
        
        # 1. Generate the Causal Attention Mask
        # Shape: (seq_len, seq_len)
        causal_mask = self.generate_causal_mask(seq_len, device)
        
        # 2. Create Positional Indices (0, 1, ..., seq_len-1)
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        
        # 3. Combine Token and Positional Embeddings
        # The network adds the concept of "meaning" (token) to "location" (position)
        x_emb = self.token_embedding(x) + self.pos_embedding(positions)
        x_emb = self.dropout(x_emb)
        
        # 4. Pass through Transformer Blocks
        # We apply the mask here. If the network is at position 1, it cannot see position 2.
        transformer_out = self.transformer(x_emb, mask=causal_mask, is_causal=True)
        
        # 5. Project to Vocabulary Space
        logits = self.fc_out(transformer_out)
        
        return logits, None
