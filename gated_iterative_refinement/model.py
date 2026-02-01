import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Refiner(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, x):
        return self.net(torch.cat([h, x], dim=-1))

class GIRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_steps, strategy='none'):
        super().__init__()
        self.num_steps = num_steps
        self.strategy = strategy
        self.num_classes = num_classes

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.refiner = Refiner(hidden_dim, input_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        if strategy == 'learned':
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim + input_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        h = self.encoder(x)
        logits_list = []
        gates_list = []

        # Initial prediction
        logits = self.head(h)
        logits_list.append(logits)

        for t in range(self.num_steps):
            delta_h = self.refiner(h, x)

            if self.strategy == 'none':
                gate = torch.ones(x.size(0), 1, device=x.device)
            elif self.strategy == 'max_prob':
                probs = F.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1, keepdim=True)[0]
                gate = 1.0 - confidence
            elif self.strategy == 'entropy':
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1, keepdim=True)
                gate = entropy / np.log(self.num_classes)
            elif self.strategy == 'learned':
                gate = self.gate_net(torch.cat([h, x], dim=-1))
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            h = h + gate * delta_h
            logits = self.head(h)
            logits_list.append(logits)
            gates_list.append(gate)

        return logits_list, gates_list

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        return [logits], []

class GRUBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        logits_list = []

        # We treat the input x as a constant input to the GRU at each step
        for _ in range(self.num_steps + 1):
            logits = self.head(h)
            logits_list.append(logits)
            h = self.gru_cell(x, h)

        return logits_list, []
