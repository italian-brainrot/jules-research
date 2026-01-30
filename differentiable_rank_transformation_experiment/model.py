import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableBatchRank(nn.Module):
    def __init__(self, num_features, initial_alpha=10.0, learnable_alpha=True):
        super().__init__()
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.full((num_features,), float(initial_alpha)))
        else:
            self.register_buffer('alpha', torch.full((num_features,), float(initial_alpha)))

    def forward(self, x):
        # x shape: (batch_size, num_features)
        # diff: (batch_size, batch_size, num_features)
        diff = x.unsqueeze(1) - x.unsqueeze(0)

        # Apply learnable alpha
        scaled_diff = diff * self.alpha

        # Compute soft rank using sigmoid
        # ranks: (batch_size, batch_size, num_features)
        ranks = torch.sigmoid(scaled_diff)

        # Average over the batch to get the rank of each element
        # (batch_size, num_features)
        return ranks.mean(dim=1)

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, use_dbrt=False, dbrt_only=False):
        super().__init__()
        self.use_dbrt = use_dbrt
        self.dbrt_only = dbrt_only

        if use_dbrt:
            self.dbrt = DifferentiableBatchRank(input_dim)
            if dbrt_only:
                current_dim = input_dim
            else:
                current_dim = input_dim * 2
        else:
            current_dim = input_dim

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        if self.use_dbrt:
            x_dbrt = self.dbrt(x)
            if self.dbrt_only:
                x = x_dbrt
            else:
                x = torch.cat([x, x_dbrt], dim=1)

        h = self.backbone(x)
        return self.classifier(h)
