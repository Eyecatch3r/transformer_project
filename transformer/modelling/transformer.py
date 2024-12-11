import torch
import torch.nn as nn
import torch.nn.functional as F

from modelling.functional import BaseTransformerLayer, TransformerDecoderLayer
from modelling.positional_encoding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        max_len,
    ):
        """
        Initialize the Transformer model.

        Args:
        - vocab_size (int): Size of the vocabulary.
        - d_model (int): Dimensionality of the embedding layer.
        - n_heads (int): Number of heads in the multi-head attention layers.
        - num_encoder_layers (int): Number of encoder layers.
        - num_decoder_layers (int): Number of decoder layers.
        - dim_feedforward (int): Dimensionality of the feedforward layer.
        - dropout (float): Dropout probability.
        - max_len (int): Maximum length of the input sequence.
        """
        super(TransformerModel, self).__init__()

        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            BaseTransformerLayer(
                input_dim=d_model,
                num_heads=n_heads,
                feature_dim=dim_feedforward,
                dropout=dropout,
                mask_future=False,
            ) for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                input_dim=d_model,
                num_heads=n_heads,
                feature_dim=dim_feedforward,
                dropout=dropout,
            ) for _ in range(num_decoder_layers)
        ])

        # Output projection (tied with embedding weights)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.output_layer.weight = self.embedding.weight  # Weight tying

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of the Transformer model.

        Args:
        - src (torch.Tensor): Source input tensor [batch_size, src_seq_len].
        - tgt (torch.Tensor): Target input tensor [batch_size, tgt_seq_len].
        - src_mask (torch.Tensor, optional): Mask for source sequences [batch_size, src_seq_len].
        - tgt_mask (torch.Tensor, optional): Mask for target sequences [batch_size, tgt_seq_len].

        Returns:
        - torch.Tensor: Output tensor of shape [batch_size, tgt_seq_len, vocab_size].
        """
        # Embedding and positional encoding
        src_emb = self.embedding(src) * (self.d_model ** 0.5)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = self.positional_encoding(tgt_emb)

        # Encoder
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, mask=src_mask)

        # Decoder
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, self_attn_mask=tgt_mask, cross_attn_mask=src_mask)

        # Output projection
        return self.output_layer(output)

# Example usage
if __name__ == "__main__":
    model = TransformerModel(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=100,
    )

    src = torch.randint(0, 10000, (32, 20))  # Batch size of 32, sequence length of 20
    tgt = torch.randint(0, 10000, (32, 20))  # Batch size of 32, sequence length of 20
    output = model(src, tgt)
    print(output.shape)  # Expected: [32, 20, 10000]
