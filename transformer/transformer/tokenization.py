CHARS = ["a", "b"]


def tokenize(s): return [CHARS.index(c) for c in s]


def untok(tok): return CHARS[tok]


# examples:
tokenize("aabaa")  # => [0, 0, 1, 0, 0]
untok(0)  # => "a"
untok(1)  # => "b"