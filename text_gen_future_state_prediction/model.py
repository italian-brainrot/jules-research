import torch
import torch.nn as nn

class GRULM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, k_future=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Auxiliary head for future state prediction
        self.fsp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.k_future = k_future

    def forward(self, x, h=None):
        embeds = self.embedding(x)
        out, h_new = self.gru(embeds, h)
        logits = self.fc(out)
        return logits, out, h_new

def get_fsp_loss(h_states, fsp_head, k_future):
    """
    h_states: (batch, seq_len, hidden_size)
    fsp_head: MLP to predict future hidden state
    k_future: how many steps into the future to predict
    """
    batch_size, seq_len, hidden_size = h_states.shape
    if seq_len <= k_future:
        return torch.tensor(0.0, device=h_states.device)

    # We want to predict h_{t+k} from h_t
    # Input h: h[:, 0 : seq_len - k_future, :]
    # Target h: h[:, k_future : seq_len, :]

    h_input = h_states[:, :-k_future, :].reshape(-1, hidden_size)
    h_target = h_states[:, k_future:, :].reshape(-1, hidden_size).detach()

    h_pred = fsp_head(h_input)
    loss = nn.MSELoss()(h_pred, h_target)
    return loss
