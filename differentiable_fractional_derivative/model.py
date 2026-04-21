import torch
import torch.nn as nn
import torch.nn.functional as F

class FractionalDerivativeLayer(nn.Module):
    def __init__(self, num_orders=1, init_orders=None):
        super().__init__()
        if init_orders is None:
            init_orders = torch.linspace(0.1, 2.0, num_orders)
        else:
            init_orders = torch.tensor(init_orders)

        self.orders = nn.Parameter(init_orders)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, channels, length = x.shape
        ks = torch.arange(1, length, device=x.device, dtype=x.dtype)
        multipliers = (ks.unsqueeze(0) - self.orders.unsqueeze(1) - 1) / ks.unsqueeze(0)
        ones = torch.ones(self.orders.shape[0], 1, device=x.device, dtype=x.dtype)
        coeffs = torch.cat([ones, multipliers], dim=1)
        coeffs = torch.cumprod(coeffs, dim=1)

        kernel = torch.flip(coeffs, dims=[1]).unsqueeze(1)

        x_padded = F.pad(x, (length - 1, 0))

        # kernel: (num_orders, 1, length)
        num_orders = self.orders.shape[0]
        # Repeat kernel for each channel to use grouped convolution
        # kernel_repeated: (channels * num_orders, 1, length)
        kernel_repeated = kernel.repeat(channels, 1, 1)

        # Output: (batch, channels * num_orders, length)
        out = F.conv1d(x_padded, kernel_repeated, groups=channels)

        # result: (batch, channels, num_orders, length)
        result = out.view(batch_size, channels, num_orders, length)
        return result

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class DFDAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, num_orders=4, hidden_dim=256, output_dim=10):
        super().__init__()
        self.dfd = FractionalDerivativeLayer(num_orders=num_orders)
        # Augment original features with fractional derivatives
        # dfd output: (batch, 1, num_orders, input_dim) -> flatten to (batch, num_orders * input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * (num_orders + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch, input_dim)
        dfd_features = self.dfd(x) # (batch, 1, num_orders, input_dim)
        dfd_features = dfd_features.view(x.size(0), -1)
        combined = torch.cat([x, dfd_features], dim=1)
        return self.mlp(combined)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    baseline = BaselineMLP()
    dfd_net = DFDAugmentedMLP(num_orders=4)
    print(f"Baseline parameters: {count_parameters(baseline)}")
    print(f"DFDAugmentedMLP parameters: {count_parameters(dfd_net)}")
