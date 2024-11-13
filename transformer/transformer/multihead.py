import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super(PositionWiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.feed_forward(x)


# Example usage
d_model = 512  # Input and output dimensionality
d_ff = 2048  # Inner-layer dimensionality

# Instantiate the layer
position_wise_ffn = PositionWiseFeedForward(d_model, d_ff)

# Create a dummy input tensor (batch_size, seq_length, d_model)
x = torch.randn(32, 10, d_model)  # Example with batch size 32 and sequence length 10

# Pass through the feed-forward layer
output = position_wise_ffn(x)
print(output.shape)  # Expected output shape: (32, 10, d_model)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimensionality per head

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)

        # Final linear layer after concatenating heads
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # Linear projections for each head
        Q = self.q_linear(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention on all the projected vectors in each head
        attn_output, attn = self.scaled_dot_product_attention(Q, K, V)

        # Concatenate heads and apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(attn_output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Multi-head attention
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Position-wise feed-forward network
        self.position_wise_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention layer with residual connection and layer normalization
        attn_output = self.multi_head_attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))

        # Position-wise feed-forward layer with residual connection and layer normalization
        ffn_output = self.position_wise_ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


# Example usage
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

# Instantiate the layer
encoder_layer = TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

# Create a dummy input tensor (batch_size, seq_length, d_model)
x = torch.randn(32, 10, d_model)  # Example with batch size 32 and sequence length 10

# Pass through the encoder layer
output = encoder_layer(x)
print(output.shape)  # Expected output shape: (32, 10, d_model)
