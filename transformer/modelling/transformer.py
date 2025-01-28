import random

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers.implementations import CharBPETokenizer
from tokenizers import Tokenizer
from transformers import AutoTokenizer, T5Tokenizer
import torch.nn.functional as F
from modelling.functional import BaseTransformerLayer, TransformerDecoderLayer
from modelling.positional_encoding import PositionalEncoding
import math


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
        # Embedding and positional encoding
        src_emb = self.embedding(src) * (self.d_model ** 0.5)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = self.positional_encoding(tgt_emb)

        # Encoder
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, mask=src_mask)
            if torch.isnan(memory).any():
                raise ValueError("NaN detected in encoder output!")

        # Decoder
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, self_attn_mask=src_mask, cross_attn_mask=tgt_mask)
            if torch.isnan(output).any():
                raise ValueError("NaN detected in decoder output!")

        # Output layer
        logits = self.output_layer(output)
        if torch.isnan(logits).any():
            raise ValueError("NaN detected in output layer!")

        return logits

    def resize_token_embeddings(self, new_vocab_size):
        """
        Resize the embedding layer to match the new vocabulary size.
        Args:
            new_vocab_size (int): The updated vocabulary size.
        """
        old_embeddings = self.embedding
        old_vocab_size, embedding_dim = old_embeddings.weight.size()

        # Create a new embedding layer with the updated size
        self.embedding = nn.Embedding(new_vocab_size, embedding_dim)

        # Copy weights from the old embeddings to the new ones
        with torch.no_grad():
            self.embedding.weight[:old_vocab_size] = old_embeddings.weight

        # Retie the output layer weights (if weight tying is used)
        self.output_layer = nn.Linear(embedding_dim, new_vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)


def autoregressive_generate(
        model,
        src_text,
        tokenizer,
        max_len=50,
        start_token="<s>",
        end_token="</s>",
        device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Performs autoregressive text generation using T5Tokenizer.

    Args:
    - model: The trained Transformer model.
    - src_text (str): Source text for translation.
    - tokenizer: T5Tokenizer for tokenization and detokenization.
    - max_len (int): Maximum length of the generated sequence.
    - start_token (str): Start token for decoding.
    - end_token (str): End token for decoding.
    - device (str): Device for computation.

    Returns:
    - str: The generated translation.
    """
    model.eval()
    model.to(device)

    # Tokenize the source text
    tokenized_input = tokenizer(
        src_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    src_tensor = tokenized_input["input_ids"]
    attention_mask = tokenized_input["attention_mask"]

    # Add <s> token for decoding
    start_token_id = tokenizer.convert_tokens_to_ids(start_token)
    end_token_id = tokenizer.convert_tokens_to_ids(end_token)

    if start_token_id is None or end_token_id is None:
        raise ValueError(f"Start or End token not found in tokenizer's vocabulary!")

    ys = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

    # Encode the source text
    src_emb = model.embedding(src_tensor) * (model.d_model ** 0.5)
    src_emb = model.positional_encoding(src_emb)

    memory = src_emb
    for layer in model.encoder_layers:
        memory = layer(memory, mask=attention_mask.unsqueeze(1).unsqueeze(2))

    # Generate tokens autoregressively
    for _ in range(max_len):
        tgt_emb = model.embedding(ys) * (model.d_model ** 0.5)
        tgt_emb = model.positional_encoding(tgt_emb)
        seq_len = ys.size(1)


        # Expand the mask to match the attention scores
        num_heads = model.decoder_layers[0].self_attention.num_heads
        tgt_mask = torch.triu(torch.ones(1, num_heads, seq_len, 128, device=device), diagonal=1).bool()
        output = tgt_emb
        for layer in model.decoder_layers:
            output = layer(output, memory, cross_attn_mask=tgt_mask)

        logits = (output[:, -1])
        probs = F.softmax(logits, dim=-1)

        # Greedy decoding
        next_token_id = torch.argmax(probs, dim=-1).item()

        ys = torch.cat([ys, torch.tensor([[next_token_id]], device=device)], dim=1)

        if next_token_id == end_token_id:
            break

    # Decode tokens back to text
    generated_tokens = ys.squeeze().tolist()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Initialize the model
    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,  # Model size
        n_heads=4,
        num_encoder_layers=4,  # Encoder layers
        num_decoder_layers=4,  # Decoder layers
        dim_feedforward=128,  # Feedforward dimension
        dropout=0.2,
        max_len=128,
    )
    model.load_state_dict(torch.load("../training/best_transformer_model.pth"))
    model.eval()

    # Load dataset
    dataset = load_dataset("wmt16", "de-en")["train"]  # Adjust dataset as needed

    # Select 10 random sentences
    random_indices = random.sample(range(len(dataset)), 10)
    random_sentences = [dataset[idx]["translation"]["de"] for idx in random_indices]

    # Translate each sentence
    for i, src_text in enumerate(random_sentences, 1):
        translation = autoregressive_generate(
            model,
            src_text,
            tokenizer,
            max_len=50,
            start_token="<s>",
            end_token="</s>",
        )
        print(f"Sentence {i}:")
        print(f"  Source: {src_text}")
        print(f"  Translation: {translation}")
        print()



