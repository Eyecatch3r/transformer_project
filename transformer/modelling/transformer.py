import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers.implementations import CharBPETokenizer
from transformers import AutoTokenizer
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import attention
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
        src_emb = self.embedding(src) * (self.d_model ** 0.5)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = self.positional_encoding(tgt_emb)

        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, mask=src_mask)
            if torch.isnan(memory).any():
                raise ValueError("NaN detected in encoder output!")

        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, self_attn_mask=tgt_mask, cross_attn_mask=src_mask)
            if torch.isnan(output).any():
                raise ValueError("NaN detected in decoder output!")

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
        src_tokens,
        tokenizer,
        max_len=50,
        start_token=None,
        end_token=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        attention_mask=None,
        temperature=1.0,
        top_k=50,
):
    model.eval()
    model.cuda() if torch.cuda.is_available() else model.cpu()
    # Token IDs for <start> and <end>
    start_token_id = tokenizer.bos_token_id if start_token is None else tokenizer.convert_tokens_to_ids(start_token)
    end_token_id = tokenizer.eos_token_id if end_token is None else tokenizer.convert_tokens_to_ids(end_token)

    if start_token_id is None or end_token_id is None:
        raise ValueError("Start or End tokens are not defined in the tokenizer vocabulary.")

    src = torch.tensor([src_tokens], dtype=torch.long, device=device)
    ys = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

    src_emb = model.embedding(src) * (model.d_model ** 0.5)
    src_emb = model.positional_encoding(src_emb)

    memory = src_emb
    for layer in model.encoder_layers:
        memory = layer(memory, mask=attention_mask)

    for _ in range(max_len):
        tgt_emb = model.embedding(ys) * (model.d_model ** 0.5)
        tgt_emb = model.positional_encoding(tgt_emb)

        seq_len = ys.size(1)
        batch_size = ys.size(0)
        num_heads = model.decoder_layers[0].self_attention.num_heads

        if seq_len == 1:
            tgt_mask = None
        else:
            # Generate causal mask for the target sequence
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            # Expand for batch size and number of heads
            tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, seq_len)

        output = tgt_emb
        for layer in model.decoder_layers:
            output = layer(output, memory, cross_attn_mask=tgt_mask, self_attn_mask=attention_mask)

        logits = (output[:, -1])
        logits = logits / temperature  # Apply temperature scaling

        # Greedy decoding: select token with highest probability
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.argmax(probs, dim=-1).item()

        # Append the selected token to the sequence
        ys = torch.cat([ys, torch.tensor([[next_token_id]], device=device)], dim=1)

        if next_token_id == end_token_id:
            break

        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        ys = torch.cat([ys, next_token_tensor], dim=1)

    generated_tokens = ys.squeeze().tolist()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    # Load dataset and tokenizer
    dataset = load_dataset("wmt/wmt17", "de-en")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Add special tokens to tokenizer
    tokenizer.add_special_tokens({"bos_token": "<start>", "eos_token": "<end>"})

    # Initialize model
    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        max_len=128,
    )

    # Resize model embeddings to account for special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Load trained weights
    model.load_state_dict(torch.load("../training/trained_transformer_model.pth"))

    # Prepare sample input
    sample = dataset["test"][0]["translation"]["de"]
    input_text = f"{tokenizer.bos_token} {sample} {tokenizer.eos_token}"
    tokenized = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)


    # Move input to device
    src_tokens = tokenized["input_ids"].squeeze().tolist()
    attention_mask = tokenized["attention_mask"].to("cuda")

    # Generate translation
    translation = autoregressive_generate(
        model,
        src_tokens,
        tokenizer,
        max_len=50,
        start_token=tokenizer.bos_token,
        end_token=tokenizer.eos_token,
        attention_mask=attention_mask,
    )

    print("Source:", sample)
    print("Generated Translation:", translation)

