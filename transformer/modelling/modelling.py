import numpy as np
import torch
import torch.nn.functional as F


def Attention(q, k, v, mask=None, mask_future=True):
    """
    Apply the attention mechanism.

    Args:
    - q (torch.Tensor): Query matrix [batch_size, n_q, d_k]
    - k (torch.Tensor): Key matrix [batch_size, n_k, d_k]
    - v (torch.Tensor): Value matrix [batch_size, n_k, d_v]
    - mask (torch.Tensor, optional): Attention mask [batch_size, n_q, n_k], where 0 values block attention.
    - mask_future (bool): Whether to apply causal masking to hide future tokens (default is True).

    Returns:
    - torch.Tensor: Attention output [batch_size, n_q, d_v]
    """
    d_k = q.size(-1)  # Dimensionality of the key/query vectors

    # Compute scaled dot-product attention scores: [batch_size, n_q, n_k]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply future masking if specified
    if mask_future:
        seq_len = attn_scores.size(-1)
        future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(attn_scores.device)
        attn_scores = attn_scores.masked_fill(future_mask, float('-inf'))

    # Apply the provided mask (e.g., padding mask) if available
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

    # Calculate attention weights using softmax
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Compute the final output by multiplying the attention weights with values: [batch_size, n_q, d_v]
    output = torch.matmul(attn_weights, v)

    return output

class Attention:
    mask = torch.tensor([[1, 0], [1, 1]])

    def __init__(self, mask_future=True):
        """
        Initialize the Attention class.

        Args:
        - causal_mask: Whether to apply causal masking to hide future tokens (default is True).
        """
        self.causal_mask = mask_future  # Default causal mask flag

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def __call__(self, q, k, v, future_mask=None):
        """
        Apply the attention mechanism.

        Args:
        - q: Query matrix [n_q, d_k]
        - k: Key matrix [n_k, d_k]
        - v: Value matrix [n_k, d_v]
        - mask: Attention mask (optional) [n_q, n_k], where large negative values block attention.

        Returns:
        - Attention output: [n_q, d_v]
        """
        d_k = q.shape[-1]

        # Compute attention scores: [n_q, n_k]
        attn_scores = q @ k.T / np.sqrt(d_k)

        # Apply the mask if provided
        if future_mask is not None:
            attn_scores += self.mask  # Mask directly added to attention scores

        # Apply softmax to get attention weights
        attn_weights = self.softmax(attn_scores)

        # Compute the final output by multiplying the attention weights with values: [n_q, d_v]
        return attn_weights @ v

# Example usage (if you want to run manually):
# q = torch.randn(2, 3, 4)  # Example batch of query tensors
# k = torch.randn(2, 3, 4)  # Example batch of key tensors
# v = torch.randn(2, 3, 4)  # Example batch of value tensors
# mask = torch.tensor([[[1, 0, 1], [1, 1, 0], [1, 1, 1]], [[1, 1, 1], [1, 0, 1], [1, 1, 0]]])  # Example mask tensor
# output = Attention(q, k, v, mask=mask, mask_future=False)
# print(output)
