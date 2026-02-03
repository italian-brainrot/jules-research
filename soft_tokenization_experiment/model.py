import torch
import torch.nn as nn
from .layers import BoundaryGuidedSoftTokenization, AttentionPooling, UniformPooling

class TokenizationModel(nn.Module):
    def __init__(self, vocab_size, input_len, output_len, dim, nhead, num_layers, method="BGST"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

        if method == "BGST":
            self.pooling = BoundaryGuidedSoftTokenization(input_len, output_len, dim)
        elif method == "Attention":
            self.pooling = AttentionPooling(output_len, dim, nhead)
        elif method == "Uniform":
            self.pooling = UniformPooling(input_len, output_len)
        else:
            raise ValueError(f"Unknown method: {method}")

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_len * dim, 4) # 4 classes for 20newsgroups subset
        )

    def forward(self, x):
        # x: (B, L) - discrete token indices
        x = self.embedding(x) # (B, L, D)
        x = self.pooling(x) # (B, M, D)
        x = self.transformer(x) # (B, M, D)
        logits = self.classifier(x) # (B, 4)
        return logits
