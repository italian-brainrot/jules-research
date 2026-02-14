import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GRDMLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, output_size=10):
        super(GRDMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.dropout1_p = 0.0
        self.dropout2_p = 0.0

    def set_dropout_rates(self, p1, p2):
        self.dropout1_p = p1
        self.dropout2_p = p2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout1_p, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout2_p, training=self.training)
        x = self.fc3(x)
        return x

class NGRDMLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, output_size=10):
        super(NGRDMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.register_buffer('p1', torch.zeros(hidden_size))
        self.register_buffer('p2', torch.zeros(hidden_size))

    def set_neuron_dropout_rates(self, p1, p2):
        self.p1.copy_(p1)
        self.p2.copy_(p2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.training:
            mask1 = torch.bernoulli(1 - self.p1).to(x.device)
            x = x * mask1 / (1 - self.p1 + 1e-8)
        x = F.relu(self.fc2(x))
        if self.training:
            mask2 = torch.bernoulli(1 - self.p2).to(x.device)
            x = x * mask2 / (1 - self.p2 + 1e-8)
        x = self.fc3(x)
        return x

class BaselineMLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, output_size=10, p=0.0):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.p = p

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.fc3(x)
        return x
