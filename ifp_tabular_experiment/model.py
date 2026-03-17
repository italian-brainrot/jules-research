import torch
import torch.nn as nn

class RIFPLayer(nn.Module):
    """
    Residual Independent Feature Preprocessing (RIFP) Layer.
    Applies a small MLP to each feature independently and adds it back to the original feature.
    """
    def __init__(self, num_features, hidden_dim=8):
        super(RIFPLayer, self).__init__()
        self.num_features = num_features
        # We use a 1D convolution with groups=num_features to apply independent MLPs efficiently.
        # Layer 1: 1 -> hidden_dim
        self.conv1 = nn.Conv1d(num_features, num_features * hidden_dim, kernel_size=1, groups=num_features)
        self.relu = nn.ReLU()
        # Layer 2: hidden_dim -> 1
        self.conv2 = nn.Conv1d(num_features * hidden_dim, num_features, kernel_size=1, groups=num_features)

    def forward(self, x):
        # x shape: (batch_size, num_features)
        residual = x
        x = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.squeeze(-1)  # (batch_size, num_features)
        return residual + x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MLP, self).__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RIFP_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, rifp_hidden_dim=8):
        super(RIFP_MLP, self).__init__()
        self.rifp = RIFPLayer(input_dim, hidden_dim=rifp_hidden_dim)
        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers)

    def forward(self, x):
        x = self.rifp(x)
        x = self.mlp(x)
        return x
