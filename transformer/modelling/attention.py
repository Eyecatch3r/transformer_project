import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Attention:
    def __init__(self, mask_future=True):
        """
        Initialize the Attention class.

        Args:
        - mask_future (bool): Whether to apply causal masking to hide future tokens (default is True).
        """
        self.mask_future = mask_future

    def __call__(self, q, k, v, mask=None):
        """
        Apply the attention mechanism.

        Args:
        - q (torch.Tensor): Query matrix [batch_size, n_q, d_k]
        - k (torch.Tensor): Key matrix [batch_size, n_k, d_k]
        - v (torch.Tensor): Value matrix [batch_size, n_k, d_v]
        - mask (torch.Tensor, optional): Attention mask [batch_size, n_q, n_k] or [batch_size, n_k]

        Returns:
        - torch.Tensor: Attention output [batch_size, n_q, d_v]
        """
        d_k = q.size(-1)  # Dimensionality of the key/query vectors

        # Compute scaled dot-product attention scores: [batch_size, n_q, n_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # Apply future (causal) mask if specified
        if self.mask_future:
            seq_len = attn_scores.size(-1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(attn_scores.device)
            attn_scores = attn_scores.masked_fill(causal_mask == 1, float('-inf'))

        # Apply provided mask (like QUERY_ATTENTION_MASK or VALUE_ATTENTION_MASK)
        if mask is not None:
            # Adjust mask shape if necessary for broadcasting
            if mask.dim() == 2:  # If mask is [batch_size, n_k]
                mask = mask.unsqueeze(1)  # Reshape to [batch_size, 1, n_k] for broadcasting
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Calculate attention weights using softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute the final output by multiplying the attention weights with values: [batch_size, n_q, d_v]
        output = torch.matmul(attn_weights, v)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mask_future=True):
        """
        Initialize the MultiHeadAttention class.

        Args:
        - d_model (int): The dimensionality of the model (input and output dimension).
        - num_heads (int): The number of attention heads.
        - mask_future (bool): Whether to apply causal masking to hide future tokens.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.mask_future = mask_future

        # Define linear transformations with biases for compatibility with state_dict
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)

        # Define the output linear transformation with bias
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k) and transpose the result
        for multi-head attention.

        Args:
        - x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]
        - batch_size (int): Batch size.

        Returns:
        - torch.Tensor: Reshaped tensor [batch_size, num_heads, seq_len, d_k]
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x, batch_size):
        """
        Combine the multi-head attention outputs into a single tensor.

        Args:
        - x (torch.Tensor): Multi-head output tensor [batch_size, num_heads, seq_len, d_k]

        Returns:
        - torch.Tensor: Combined tensor [batch_size, seq_len, d_model]
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        q = self.split_heads(self.query_transform(q), batch_size)
        k = self.split_heads(self.key_transform(k), batch_size)
        v = self.split_heads(self.value_transform(v), batch_size)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask if needed
        if self.mask_future:
            seq_len_q, seq_len_k = attn_scores.size(-2), attn_scores.size(-1)
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k), diagonal=1).to(attn_scores.device)
            attn_scores = attn_scores.masked_fill(causal_mask == 1, float(-1e9))

        if mask is not None:
            # Ensure mask has correct dimensions: [batch_size, num_heads, query_len, key_len]
            if mask.dim() == 2:  # Mask is [batch_size, key_len]
                mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, key_len]
            elif mask.dim() == 3:  # Mask is [batch_size, query_len, key_len]
                mask = mask.unsqueeze(1)  # Shape: [batch_size, 1, query_len, key_len]

            # Expand mask to match attention scores' dimensions
            query_len = attn_scores.size(-2)
            key_len = attn_scores.size(-1)
            batch_size, num_heads = attn_scores.size(0), attn_scores.size(1)

            # Align mask dimensions
            mask = mask.expand(batch_size, num_heads, query_len, key_len)

            # Debug shape alignment
            if mask.size() != attn_scores.size():
                raise ValueError(
                    f"Mask and attention scores size mismatch: mask={mask.size()}, attn_scores={attn_scores.size()}"
                )

            # Apply mask
            attn_scores = attn_scores.masked_fill(mask == 0, float(-1e9))

        # Softmax and attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Combine heads and final transformation
        output = self.combine_heads(output, batch_size)
        return self.output_transform(output)

def create_padding_mask(seq, pad_token_id):
    """
    Create a mask for padding tokens in the sequence.

    Args:
    - seq (torch.Tensor): Input sequence tensor [batch_size, seq_len].
    - pad_token_id (int): Padding token ID in the vocabulary.

    Returns:
    - mask (torch.Tensor): Padding mask [batch_size, 1, 1, seq_len].
    """
    # Mask positions where the value is equal to `pad_token_id`
    mask = (seq == pad_token_id).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    return mask

def test_padding_mask():
    """
    Test function to verify if masked padding correctly identifies and masks padding tokens.
    """
    # Example sequence batch with padding tokens (e.g., 0 is the pad_token_id)
    seq = torch.tensor([
        [1, 2, 3, 0, 0],  # Padding at the end
        [4, 5, 0, 0, 0],  # More padding
        [6, 7, 8, 9, 10]  # No padding
    ])  # Shape: [batch_size, seq_len]

    pad_token_id = 0  # Define the padding token ID

    # Create the padding mask
    mask = (seq == pad_token_id).unsqueeze(1).unsqueeze(2).float()  # Shape: [batch_size, 1, 1, seq_len]

    # Example attention scores (random values for testing)
    attention_scores = torch.rand(seq.size(0), 1, seq.size(1), seq.size(1))  # Shape: [batch_size, num_heads, query_len, key_len]

    # Apply the mask to attention scores
    masked_attention_scores = attention_scores.masked_fill(mask == 1, float('-inf'))

    # Print results
    print("Input Sequence:")
    print(seq)
    print("\nGenerated Mask:")
    print(mask)
    print("\nAttention Scores Before Masking:")
    print(attention_scores)
    print("\nAttention Scores After Masking:")
    print(masked_attention_scores)