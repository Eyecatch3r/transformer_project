import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

        # Register as a buffer to avoid being a learnable parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass to add positional encoding to the input tensor.

        Args:
        - x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, d_model]

        Returns:
        - torch.Tensor: Tensor with positional encodings added, same shape as input [batch_size, sequence_length, d_model]
        """
        # Add positional encoding, matching the input sequence length
        x = x + self.pe[:, :x.size(1), :]
        return x
