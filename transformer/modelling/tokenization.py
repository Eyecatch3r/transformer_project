import re
from collections import defaultdict, Counter

from tokenizers.implementations import CharBPETokenizer


class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.bpe_rules = {}

    def get_vocab(self, corpus):
        """
        Generate vocabulary based on character-level tokens.
        """
        vocab = defaultdict(int)
        for word in corpus:
            # Add a space between characters to simulate character-level tokenization
            chars = " ".join(list(word)) + " </w>"
            vocab[chars] += 1
        return vocab

    def get_stats(self, vocab):
        """
        Get statistics of pairs of symbols in the vocabulary.
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """
        Merge the most frequent pair in the vocabulary.
        """
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_vocab = {}
        for word in vocab:
            # Merge the frequent bigram in the vocabulary
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def fit(self, corpus):
        """
        Train the BPE tokenizer on a given corpus.
        """
        vocab = self.get_vocab(corpus)
        for i in range(self.vocab_size):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.bpe_rules[best] = i

        # The final vocabulary
        self.vocab = {word.replace(' ', ''): freq for word, freq in vocab.items()}

    def tokenize(self, word):
        """
        Tokenize a word based on the learned BPE rules.
        """
        word = list(word) + ['</w>']
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            candidate_pairs = [pair for pair in pairs if pair in self.bpe_rules]
            if not candidate_pairs:
                break
            best_pair = min(candidate_pairs, key=lambda x: self.bpe_rules[x])
            word = [best_pair[0] + best_pair[1] if (word[i], word[i + 1]) == best_pair else word[i] for i in
                    range(len(word) - 1)]
        return word


tokenizer = CharBPETokenizer()

# Example Usage
corpus = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning."
]

# Write the corpus into a text file
file_path = 'corpus.txt'
"""
with open(file_path, 'w') as file:
    for sentence in corpus:
        file.write(sentence + '\n')
"""

tokenizer_own = BPETokenizer(vocab_size=64)
tokenizer.train(file_path)
tokenizer_own.fit(corpus)

# print("Vocabulary: ", tokenizer.vocab)
# print("Tokenized 'newest': ", tokenizer.tokenize('machine'))

encoded_BPE = tokenizer.encode("Machine learning is a subset of artificial intelligence.")
encoded_own = tokenizer_own.tokenize("Machine learning is a subset of artificial intelligence.")
print("Huggingface: ",encoded_BPE.tokens)
print("Own: ", encoded_own)