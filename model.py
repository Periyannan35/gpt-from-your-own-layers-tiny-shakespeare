"""
GPT From Your Own Layers: Tiny Shakespeare

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import os
import tempfile
import urllib.request
import torch

def load_data(train_frac=0.9):
    """Return (chars, encode, decode, train_data, val_data) for Tiny Shakespeare, one id per character."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = os.path.join(tempfile.gettempdir(), "tinyshakespeare.txt")

    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join([itos[i] for i in ids])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return chars, encode, decode, train_data, val_data

# Step 2 - get_batch (not yet solved)
# TODO: implement

# Step 3 - BigramLanguageModel (not yet solved)
# TODO: implement

# Step 4 - generate (not yet solved)
# TODO: implement

# Step 5 - train (not yet solved)
# TODO: implement

# Step 6 - estimate_loss (not yet solved)
# TODO: implement

# Step 7 - average_past (not yet solved)
# TODO: implement

# Step 8 - Head (not yet solved)
# TODO: implement

# Step 9 - MultiHeadAttention (not yet solved)
# TODO: implement

# Step 10 - FeedForward (not yet solved)
# TODO: implement

# Step 11 - LayerNorm (not yet solved)
# TODO: implement

# Step 12 - Block (not yet solved)
# TODO: implement

# Step 13 - GPTLanguageModel (not yet solved)
# TODO: implement

