from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from modelling.attention import MultiHeadAttention

class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim=512, num_heads=8, feature_dim=2048, dropout=0.1, mask_future=False):
        """
        Base Transformer Layer with Multi-Head Attention and Position-Wise Feed-Forward Network.

        Args:
        - input_dim (int): Model dimension.
        - num_heads (int): Number of attention heads.
        - feature_dim (int): Feed-forward network dimension.
        - dropout (float): Dropout rate.
        - mask_future (bool): Whether to apply causal masking to hide future tokens.
        """
        super(BaseTransformerLayer, self).__init__()

        # Multi-head attention layer
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future)

        # Layer normalization layers
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        # Define position-wise feed-forward network layers explicitly
        self.feature_transformation = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, feature_dim, bias=True)),
            ('relu', nn.ReLU()),
            ('linear2', nn.Linear(feature_dim, input_dim, bias=True))
        ]))

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for the Transformer Layer with Attention Mask.

        Args:
        - x (torch.Tensor): Input tensor [batch_size, seq_len, input_dim].
        - mask (torch.Tensor, optional): Attention mask [batch_size, seq_len].

        Returns:
        - torch.Tensor: Output tensor after applying multi-head attention and feed-forward network.
        """

        # Multi-head attention with optional mask
        attn_output = self.self_attention(x, x, x, mask)

        # Residual connection, dropout, and layer normalization
        x_residual_1 = x + self.dropout1(attn_output)

        x = self.layer_norm_1(x_residual_1)

        # Position-wise feed-forward network
        ffn_output = self.feature_transformation(x)

        # Residual connection, dropout, and second layer normalization
        x_residual_2 = x + self.dropout2(ffn_output)

        x = self.layer_norm_2(x_residual_2)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim=512, num_heads=8, feature_dim=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # Self-attention (with future masking)
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True)

        # Cross-attention (encoder-decoder attention)
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False)

        # Feed-forward network with two linear transformations
        self.feature_transformation = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, feature_dim)),
            ('relu', nn.ReLU()),
            ('linear2', nn.Linear(feature_dim, input_dim))
        ]))

        # Layer normalization layers
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # Self-attention
        self_attn_output = self.self_attention(x, x, x, mask=self_attn_mask)

        # Add and norm
        x = self.layer_norm_1(x + self.dropout1(self_attn_output))

        # Cross-attention
        cross_attn_output = self.encoder_attention(x, encoder_output, encoder_output, mask=cross_attn_mask)

        # Add and norm
        x = self.layer_norm_2(x + self.dropout2(cross_attn_output))

        # Feedforward
        feedforward_output = self.feature_transformation(x)

        # Add and norm
        x = self.layer_norm_3(x + self.dropout3(feedforward_output))

        return x

