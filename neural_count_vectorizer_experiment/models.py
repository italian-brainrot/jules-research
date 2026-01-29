import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightingScheme(nn.Module):
    def forward(self, tf, df, doc_len, avg_doc_len, num_docs):
        raise NotImplementedError

class UniformWeighting(WeightingScheme):
    def forward(self, tf, df, doc_len, avg_doc_len, num_docs):
        return torch.ones_like(tf)

class TFWeighting(WeightingScheme):
    def forward(self, tf, df, doc_len, avg_doc_len, num_docs):
        return tf

class LogTFIDFWeighting(WeightingScheme):
    def forward(self, tf, df, doc_len, avg_doc_len, num_docs):
        idf = torch.log(num_docs / (df + 1e-6) + 1.0)
        return torch.log1p(tf) * idf

class BM25Weighting(WeightingScheme):
    def __init__(self, k1=1.5, b=0.75):
        super().__init__()
        self.k1 = nn.Parameter(torch.tensor(k1), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=False)

    def forward(self, tf, df, doc_len, avg_doc_len, num_docs):
        idf = torch.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        return idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len))

class NCVWeighting(WeightingScheme):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, tf, df, doc_len, avg_doc_len, num_docs):
        # Inputs: log(1+tf), log(1+df), log(doc_len/avg_doc_len + 1)
        x = torch.stack([
            torch.log1p(tf),
            torch.log1p(df),
            torch.log(doc_len / avg_doc_len + 1.0)
        ], dim=-1)
        return self.mlp(x).squeeze(-1)

class BOWClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim, weighting_scheme, df, avg_doc_len, num_docs):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.weighting_scheme = weighting_scheme
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.register_buffer('df', df)
        self.avg_doc_len = avg_doc_len
        self.num_docs = num_docs

    def forward(self, batch):
        device = self.df.device
        all_indices = []
        all_tf = []
        all_doc_len = []
        batch_map = []

        for i, (indices, tf, doc_len, _) in enumerate(batch):
            all_indices.append(indices)
            all_tf.append(tf)
            # doc_len is a scalar tensor
            all_doc_len.append(torch.full_like(tf, doc_len))
            batch_map.append(torch.full_like(tf, i, dtype=torch.long))

        if not all_indices:
            # Handle empty batch if necessary
            return None

        all_indices = torch.cat(all_indices).to(device)
        all_tf = torch.cat(all_tf).to(device)
        all_doc_len = torch.cat(all_doc_len).to(device)
        batch_map = torch.cat(batch_map).to(device)

        df_i = self.df[all_indices]
        weights = self.weighting_scheme(all_tf, df_i, all_doc_len, self.avg_doc_len, self.num_docs)

        e_i = self.embeddings(all_indices)
        weighted_embeddings = weights.unsqueeze(-1) * e_i

        embedding_dim = e_i.shape[1]
        num_docs_in_batch = len(batch)
        doc_vectors = torch.zeros(num_docs_in_batch, embedding_dim, device=device)
        doc_vectors.scatter_add_(0, batch_map.unsqueeze(-1).expand(-1, embedding_dim), weighted_embeddings)

        logits = self.classifier(doc_vectors)
        return logits
