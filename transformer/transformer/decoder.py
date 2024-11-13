import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # Self-attention layer with future masking
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # Cross-attention layer for encoder-decoder attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # Self-attention with residual connection and layer normalization
        _x = x
        x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = _x + self.dropout(x)
        x = self.norm1(x)

        # Cross-attention with residual connection and layer normalization
        _x = x
        x, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = _x + self.dropout(x)
        x = self.norm2(x)

        # Feed-forward layer with residual connection and layer normalization
        _x = x
        x = self.feed_forward(x)
        x = _x + self.dropout(x)
        x = self.norm3(x)

        return x
