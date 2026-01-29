import urllib.request
import os
import torch
import numpy as np

def get_data(subset_size=100000):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")

    if not os.path.exists(cache_path):
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
        with open(cache_path, "w") as f:
            f.write(text)
    else:
        with open(cache_path, "r") as f:
            text = f.read()

    text = text[:subset_size]
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    data = [char_to_ix[ch] for ch in text]
    return torch.tensor(data, dtype=torch.long), vocab_size, char_to_ix, ix_to_char

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index : index + self.seq_length]
        y = self.data[index + 1 : index + self.seq_length + 1]
        return x, y

def get_dataloaders(batch_size, seq_length, subset_size=100000):
    data, vocab_size, char_to_ix, ix_to_char = get_data(subset_size)
    n = len(data)
    train_data = data[:int(n*0.8)]
    val_data = data[int(n*0.8):]

    train_ds = CharDataset(train_data, seq_length)
    val_ds = CharDataset(val_data, seq_length)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, vocab_size, char_to_ix, ix_to_char
