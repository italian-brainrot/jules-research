import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftBinningLayer(nn.Module):
    def __init__(self, num_features, num_bins, temperature=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        # Centers for each bin for each feature
        # Shape: (num_features, num_bins)
        # Initialize centers uniformly across the expected range [-5, 5]
        self.centers = nn.Parameter(torch.linspace(-5, 5, num_bins).repeat(num_features, 1))
        self.log_temperature = nn.Parameter(torch.full((num_features,), torch.log(torch.tensor(temperature))))

    def forward(self, x):
        # x shape: (batch, num_features)
        x_unsqueezed = x.unsqueeze(-1)
        # centers shape: (num_features, num_bins)
        dist = torch.pow(x_unsqueezed - self.centers, 2)

        temp = torch.exp(self.log_temperature).view(1, self.num_features, 1)
        logits = -dist / (temp + 1e-6)
        soft_bins = F.softmax(logits, dim=-1)

        return soft_bins.view(x.size(0), -1)

class ResidualSoftBinMLP(nn.Module):
    def __init__(self, input_dim=40, num_bins=8, hidden_dim=256, output_dim=10):
        super().__init__()
        self.soft_binning = SoftBinningLayer(input_dim, num_bins)
        # Project binned features back to a manageable size or use them directly
        # Let's try to mix them with original features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * num_bins + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        binned = self.soft_binning(x)
        combined = torch.cat([x, binned], dim=1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim1=512, hidden_dim2=256, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    rsb_mlp = ResidualSoftBinMLP(num_bins=8)
    base_mlp = BaselineMLP()
    print(f"ResidualSoftBinMLP parameters: {count_parameters(rsb_mlp)}")
    print(f"BaselineMLP parameters: {count_parameters(base_mlp)}")
