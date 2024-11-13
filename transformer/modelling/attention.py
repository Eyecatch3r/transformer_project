# Model ops from https://github.com/jaymody/picoGPT/blob/main/gpt2.py (MIT license)
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# [m, in], [in, out], [out] -> [m, out]
def linear(x, w, b):
    return x @ w + b


# [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


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
        - mask_future (bool): Whether to apply causal masking to hide future tokens (default is True).
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
        """
        Apply the multi-head attention mechanism.

        Args:
        - q (torch.Tensor): Query matrix [batch_size, seq_len, d_model]
        - k (torch.Tensor): Key matrix [batch_size, seq_len, d_model]
        - v (torch.Tensor): Value matrix [batch_size, seq_len, d_model]
        - mask (torch.Tensor, optional): Attention mask [batch_size, seq_len] or [batch_size, 1, 1, seq_len]

        Returns:
        - torch.Tensor: Multi-head attention output [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)

        # Linear projections for query, key, and value
        q = self.query_transform(q)
        k = self.key_transform(k)
        v = self.value_transform(v)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)  # [batch_size, num_heads, seq_len, d_k]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply future (causal) mask if specified
        if self.mask_future:
            seq_len = attn_scores.size(-1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(attn_scores.device)
            attn_scores = attn_scores.masked_fill(causal_mask == 1, float('-inf'))

        # Apply provided mask if available
        if mask is not None:
            if mask.dim() == 2:  # Mask is [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)  # Reshape to [batch_size, 1, 1, seq_len] for broadcasting
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute output by weighted sum of values
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, d_k]

        # Concatenate heads and reshape back to [batch_size, seq_len, d_model]
        output = self.combine_heads(output, batch_size)

        # Final linear transformation
        output = self.output_transform(output)

        return output


# [n_seq, n_embd] -> [n_seq, n_embd]
def causal_self_attention(x, c_attn, c_proj):
    # qkv projections
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    q, k, v = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform causal self attention
    x = attention(q, k, v, causal_mask)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

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
