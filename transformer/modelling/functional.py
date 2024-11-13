from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from modelling.attention import MultiHeadAttention


# Define softmax function using numpy for compatibility
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Linear transformation
def linear(x, w, b):
    return x @ w + b


# Attention class
class Attention:
    def __init__(self, mask_future=True):
        """
        Initialize the Attention class.

        Args:
        - mask_future (bool): Whether to apply causal masking to hide future tokens (default is True).
        """
        self.mask_future = mask_future

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def __call__(self, q, k, v):
        """
        Apply the attention mechanism.

        Args:
        - q (np.array): Query matrix [n_q, d_k]
        - k (np.array): Key matrix [n_k, d_k]
        - v (np.array): Value matrix [n_k, d_v]

        Returns:
        - np.array: Attention output [n_q, d_v]
        """
        # Check dimensional compatibility
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("The dimensionality of queries and keys must match (d_k).")
        if k.shape[0] != v.shape[0]:
            raise ValueError("The number of keys (n_k) must match the number of values (n_k).")

        d_k = q.shape[-1]

        # Compute attention scores: [n_q, n_k]
        attn_scores = q @ k.T / np.sqrt(d_k)

        # Generate causal mask if specified
        if self.mask_future:
            seq_len = attn_scores.shape[-1]
            causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * -1e10
            attn_scores = np.where(causal_mask == 0, attn_scores, causal_mask)

        # Apply softmax to get attention weights
        attn_weights = self.softmax(attn_scores)

        # Compute the final output by multiplying the attention weights with values: [n_q, d_v]
        return attn_weights @ v


class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim=512, num_heads=8, feature_dim=2048, dropout=0.1, mask_future=True):
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
        x = self.layer_norm_1(x + self.dropout1(attn_output))

        # Position-wise feed-forward network with residual connection
        ffn_output = self.feature_transformation(x)

        # Dropout and second layer normalization
        x = self.layer_norm_2(x + self.dropout2(ffn_output))

        return x


# Example function for causal self-attention
def causal_self_attention(x, c_attn, c_proj):
    # Linear projection to get q, k, v
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    q, k, v = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # Initialize the Attention class with causal masking enabled
    attention_layer = Attention(mask_future=True)

    # Perform causal self-attention
    x = attention_layer(q, k, v)  # [n_seq, n_embd]

    # Output projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x


# [n_seq, n_embd] -> [n_seq, n_embd]
def transformer_block(x, attn):
    x = x + causal_self_attention(x, **attn)
    # NOTE: removed ffn
    return x


# [n_seq] -> [n_seq, n_vocab]
def gpt(inputs, wte, wpe, blocks):
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


N_CTX = 5
N_VOCAB = 2
N_EMBED = 8

Lg = 1024  # Large

MODEL = {
    # EMBEDDING USAGE
    #  P = Position embeddings (one-hot)
    #  T = Token embeddings (one-hot, first is `a`, second is `b`)
    #  V = Prediction scratch space
    #
    #       [P, P, P, P, P, T, T, V]
    "wte": np.array(
        # one-hot token embeddings
        [
            [0, 0, 0, 0, 0, 1, 0, 0],  # token `a` (id 0)
            [0, 0, 0, 0, 0, 0, 1, 0],  # token `b` (id 1)
        ]
    ),
    "wpe": np.array(
        # one-hot position embeddings
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # position 0
            [0, 1, 0, 0, 0, 0, 0, 0],  # position 1
            [0, 0, 1, 0, 0, 0, 0, 0],  # position 2
            [0, 0, 0, 1, 0, 0, 0, 0],  # position 3
            [0, 0, 0, 0, 1, 0, 0, 0],  # position 4
        ]
    ),
    "blocks": [
        {
            "attn": {
                "c_attn": {  # generates qkv matrix
                    "b": np.zeros(N_EMBED * 3),
                    "w": np.array(
                        # this is where the magic happens
                        # fmt: off
                        [
                            [Lg, 0., 0., 0., 0., 0., 0., 0.,  # q
                             1., 0., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [Lg, Lg, 0., 0., 0., 0., 0., 0.,  # q
                             0., 1., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [0., Lg, Lg, 0., 0., 0., 0., 0.,  # q
                             0., 0., 1., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [0., 0., Lg, Lg, 0., 0., 0., 0.,  # q
                             0., 0., 0., 1., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [0., 0., 0., Lg, Lg, 0., 0., 0.,  # q
                             0., 0., 0., 0., 1., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [0., 0., 0., 0., 0., 0., 0., 0.,  # q
                             0., 0., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 1.],  # v
                            [0., 0., 0., 0., 0., 0., 0., 0.,  # q
                             0., 0., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., -1],  # v
                            [0., 0., 0., 0., 0., 0., 0., 0.,  # q
                             0., 0., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                        ]
                        # fmt: on
                    ),
                },
                "c_proj": {  # weights to project attn result back to embedding space
                    "b": [0, 0, 0, 0, 0, Lg, 0, 0],
                    "w": np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, -Lg, Lg, 0],
                        ]
                    ),
                },
            },
        }
    ],
}

CHARS = ["a", "b"]


def tokenize(s): return [CHARS.index(c) for c in s]


def untok(tok): return CHARS[tok]


def predict(s):
    tokens = tokenize(s)[-5:]
    logits = gpt(np.array(tokens), **MODEL)
    probs = softmax(logits)

    for i, tok in enumerate(tokens):
        pred = np.argmax(probs[i])
        print(
            f"{untok(tok)} ({tok}): next={untok(pred)} ({pred}) probs={probs[i]} logits={logits[i]}"
        )

    return np.argmax(probs[-1])


def complete(s, max_new_tokens=10):
    tokens = tokenize(s)
    while len(tokens) < len(s) + max_new_tokens:
        logits = gpt(np.array(tokens[-5:]), **MODEL)
        probs = softmax(logits)
        pred = np.argmax(probs[-1])
        tokens.append(pred)
    return s + " :: " + "".join(untok(t) for t in tokens[len(s):])
