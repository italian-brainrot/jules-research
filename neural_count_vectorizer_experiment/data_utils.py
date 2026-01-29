import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, X_counts, y):
        self.X_counts = X_counts.tocsr()
        self.y = torch.tensor(y, dtype=torch.long)
        self.doc_lengths = torch.tensor(np.array(X_counts.sum(axis=1)).flatten(), dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        row = self.X_counts[idx]
        indices = torch.tensor(row.indices, dtype=torch.long)
        values = torch.tensor(row.data, dtype=torch.float32)
        return indices, values, self.doc_lengths[idx], self.y[idx]

def collate_fn(batch):
    # batch is list of (indices, values, doc_len, y)
    # We want to return them separately
    return batch

def get_data(categories=None, vocab_size=5000):
    if categories is None:
        categories = ['rec.autos', 'rec.motorcycles', 'sci.electronics', 'sci.space']

    print(f"Loading 20newsgroups for categories: {categories}")
    data_train_all = fetch_20newsgroups(subset='train', categories=categories,
                                        remove=('headers', 'footers', 'quotes'))
    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   remove=('headers', 'footers', 'quotes'))

    # Split train into train and val
    texts_train, texts_val, y_train, y_val = train_test_split(
        data_train_all.data, data_train_all.target, test_size=0.2, random_state=42, stratify=data_train_all.target
    )

    vectorizer = CountVectorizer(max_features=vocab_size, stop_words='english')
    X_train_counts = vectorizer.fit_transform(texts_train)
    X_val_counts = vectorizer.transform(texts_val)
    X_test_counts = vectorizer.transform(data_test.data)

    df = np.array((X_train_counts > 0).sum(axis=0)).flatten()
    doc_lengths = np.array(X_train_counts.sum(axis=1)).flatten()
    avg_doc_length = doc_lengths.mean()
    num_docs = X_train_counts.shape[0]

    datasets = {
        'train': TextDataset(X_train_counts, y_train),
        'val': TextDataset(X_val_counts, y_val),
        'test': TextDataset(X_test_counts, data_test.target)
    }

    stats = {
        'df': torch.tensor(df, dtype=torch.float32),
        'avg_doc_length': avg_doc_length,
        'num_docs': num_docs,
        'vocab_size': len(vectorizer.get_feature_names_out())
    }

    return datasets, stats

if __name__ == "__main__":
    datasets, stats = get_data()
    print(f"Train size: {len(datasets['train'])}")
    print(f"Val size: {len(datasets['val'])}")
    print(f"Test size: {len(datasets['test'])}")
    print(f"Vocab size: {stats['vocab_size']}")
    print(f"Avg doc length: {stats['avg_doc_length']:.2f}")

    # Test loader
    from torch.utils.data import DataLoader
    dl = DataLoader(datasets['train'], batch_size=4, shuffle=True, collate_fn=collate_fn)
    for batch in dl:
        for indices, values, doc_len, y in batch:
            print(f"Doc len: {doc_len}, y: {y}, num_terms: {len(indices)}")
        break
