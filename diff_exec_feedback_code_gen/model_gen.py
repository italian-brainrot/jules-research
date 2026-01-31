import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class Generator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x, target_len=5, teacher_forcing=None):
        # x: (batch, seq_len)
        batch_size = x.size(0)
        embedded = self.embedding(x)
        _, h = self.encoder(embedded) # h: (1, batch, hidden_dim)

        # Decoder input: <SOS> (index 1)
        decoder_input = torch.full((batch_size, 1), 1, device=x.device)

        logits_list = []
        for i in range(target_len):
            embedded_dec = self.embedding(decoder_input)
            output, h = self.decoder(embedded_dec, h)
            logits = self.out(output) # (batch, 1, vocab_size)
            logits_list.append(logits)

            if teacher_forcing is not None:
                decoder_input = teacher_forcing[:, i].unsqueeze(1)
            else:
                decoder_input = torch.argmax(logits, dim=-1)

        return torch.cat(logits_list, dim=1) # (batch, target_len, vocab_size)

    def forward_soft(self, x, target_len=5, tau=1.0):
        # Gumbel-softmax version for differentiable feedback
        batch_size = x.size(0)
        embedded = self.embedding(x)
        _, h = self.encoder(embedded)

        # Initial decoder input embedding (<SOS> is index 1)
        curr_emb = self.embedding(torch.full((batch_size, 1), 1, device=x.device))

        soft_tokens_list = []
        for i in range(target_len):
            output, h = self.decoder(curr_emb, h)
            logits = self.out(output).squeeze(1)

            soft_token = F.gumbel_softmax(logits, tau=tau, hard=False)
            soft_tokens_list.append(soft_token.unsqueeze(1))

            # Next embedding is a weighted sum of all embeddings (soft input)
            curr_emb = torch.matmul(soft_token, self.embedding.weight).unsqueeze(1)

        return torch.cat(soft_tokens_list, dim=1) # (batch, target_len, vocab_size)
