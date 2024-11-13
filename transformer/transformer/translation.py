import re


def clean_sentence(sentence):
    # Remove non-UTF8 characters
    sentence = sentence.encode('utf-8', 'ignore').decode('utf-8')

    # Remove URLs and HTML tags
    sentence = re.sub(r'http\S+|<.*?>', '', sentence)

    # Convert to lowercase
    sentence = sentence.lower()

    # Remove unwanted characters using the whitelist
    whitelist = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    sentence = ''.join([char for char in sentence if char in whitelist])

    # Split and trim sentences that are too short or too long
    words = sentence.split()
    if len(words) < 5 or len(words) > 64:
        return None
    return ' '.join(words)


def clean_dataset(dataset):
    cleaned_data = []
    for item in dataset:
        source = clean_sentence(item['translation']['de'])
        target = clean_sentence(item['translation']['en'])
        if source and target:
            # Skip pairs where the source-target length ratio is too large
            if 0.5 < len(source.split()) / len(target.split()) < 2.0:
                cleaned_data.append({'source': source, 'target': target})
    return cleaned_data


from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer


class CustomTokenizer:
    def __init__(self, vocab_size=50000):
        # Initialize ByteLevel BPE Tokenizer
        self.tokenizer = ByteLevelBPETokenizer()
        self.vocab_size = vocab_size

    def train_tokenizer(self, files):
        # Train the tokenizer
        self.tokenizer.train(files, vocab_size=self.vocab_size)

    def save_tokenizer(self, save_directory):
        # Save the trained tokenizer
        self.tokenizer.save_model(save_directory)

    def load_gpt2_tokenizer(self, save_directory):
        # Load the tokenizer as a GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained(save_directory)


import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data[idx]['source']
        target = self.data[idx]['target']

        # Encode the source and target sentences
        source_encoded = self.tokenizer(source, padding='max_length', truncation=True, max_length=self.max_length,
                                        return_tensors='pt')
        target_encoded = self.tokenizer(target, padding='max_length', truncation=True, max_length=self.max_length,
                                        return_tensors='pt')

        return source_encoded['input_ids'].squeeze(), target_encoded['input_ids'].squeeze()


import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register as a buffer to avoid being a learnable parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input tensor
        return x + self.pe[:, :x.size(1)]


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)
