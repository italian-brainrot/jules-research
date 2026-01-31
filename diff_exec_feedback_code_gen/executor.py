import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import random

class ProxyExecutor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, inputs):
        # x: (batch, seq_len, vocab_size) if soft, (batch, seq_len) if hard
        if x.dim() == 3:
            # Soft tokens (probability distributions)
            embedded = torch.matmul(x, self.embedding.weight)
        else:
            # Hard tokens (indices)
            embedded = self.embedding(x)

        _, h = self.gru(embedded) # h: (1, batch, hidden_dim)
        h = h.squeeze(0)

        # Concatenate with inputs (a, b, c)
        # inputs: (batch, 3)
        combined = torch.cat([h, inputs], dim=1)
        out = self.mlp(combined)
        return out.squeeze(1)

def train_executor():
    with open("diff_exec_feedback_code_gen/vocab.json", "r") as f:
        token_to_idx = json.load(f)
    with open("diff_exec_feedback_code_gen/train.json", "r") as f:
        train_data = json.load(f)

    vocab_size = len(token_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProxyExecutor(vocab_size, 64, 256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Prepare data
    prepared_data = []
    for d in train_data:
        expr = d['expression']
        expr_ids = [token_to_idx.get(t, token_to_idx['<UNK>']) for t in expr]
        for io in d['io_pairs']:
            inputs = torch.tensor([io['inputs']['a'], io['inputs']['b'], io['inputs']['c']], dtype=torch.float32)
            output = torch.tensor(io['output'], dtype=torch.float32)
            prepared_data.append((torch.tensor(expr_ids), inputs, output))

    print(f"Training executor on {len(prepared_data)} samples...")

    # Simple training loop
    batch_size = 128
    for epoch in range(50):
        random.shuffle(prepared_data)
        total_loss = 0
        for i in range(0, len(prepared_data), batch_size):
            batch = prepared_data[i:i+batch_size]
            if not batch: continue
            # Pad sequences to 5
            exprs = torch.stack([F.pad(b[0], (0, 5 - len(b[0]))) for b in batch]).to(device)
            inputs = torch.stack([b[1] for b in batch]).to(device)
            targets = torch.stack([b[2] for b in batch]).to(device)

            optimizer.zero_grad()
            outputs = model(exprs, inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(prepared_data)/batch_size)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "diff_exec_feedback_code_gen/executor.pth")
    print("Executor trained and saved.")

if __name__ == "__main__":
    train_executor()
