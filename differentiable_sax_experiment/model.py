import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableSAX(nn.Module):
    def __init__(self, num_segments=8, alphabet_size=4, temperature=10.0, learnable_breakpoints=True):
        super().__init__()
        self.num_segments = num_segments
        self.alphabet_size = alphabet_size
        self.temperature = temperature

        if alphabet_size == 2:
            initial_breakpoints = [0.0]
        elif alphabet_size == 3:
            initial_breakpoints = [-0.43, 0.43]
        elif alphabet_size == 4:
            initial_breakpoints = [-0.67, 0.0, 0.67]
        elif alphabet_size == 5:
            initial_breakpoints = [-0.84, -0.25, 0.25, 0.84]
        else:
            initial_breakpoints = torch.linspace(-1, 1, alphabet_size - 1).tolist()

        self.breakpoints = nn.Parameter(torch.tensor(initial_breakpoints), requires_grad=learnable_breakpoints)

    def forward(self, x):
        # x shape: (B, L)
        B, L = x.shape

        # 1. Z-normalization
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - mean) / std

        # 2. PAA
        x_norm = x_norm.unsqueeze(1) # (B, 1, L)
        paa = F.adaptive_avg_pool1d(x_norm, self.num_segments) # (B, 1, num_segments)
        paa = paa.squeeze(1) # (B, num_segments)

        # 3. Soft Quantization
        sorted_breakpoints = torch.sort(self.breakpoints)[0]
        s_minus_b = paa.unsqueeze(-1) - sorted_breakpoints.unsqueeze(0).unsqueeze(0)
        sigmoids = torch.sigmoid(self.temperature * s_minus_b)

        probs = []
        # Bin 0
        probs.append(1.0 - sigmoids[:, :, 0:1])
        # Bins 1 to alphabet_size - 2
        for i in range(self.alphabet_size - 2):
            probs.append(sigmoids[:, :, i:i+1] - sigmoids[:, :, i+1:i+2])
        # Last bin
        probs.append(sigmoids[:, :, -1:])

        # (B, num_segments, alphabet_size)
        sax_soft = torch.cat(probs, dim=-1)

        return sax_soft

class SAXAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, num_segments=8, alphabet_size=4):
        super().__init__()
        self.sax_layer = DifferentiableSAX(num_segments=num_segments, alphabet_size=alphabet_size)
        sax_out_dim = num_segments * alphabet_size

        self.mlp = nn.Sequential(
            nn.Linear(input_dim + sax_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (B, L)
        sax_features = self.sax_layer(x)
        sax_features = sax_features.view(x.shape[0], -1)
        combined = torch.cat([x, sax_features], dim=1)
        return self.mlp(combined)

class SAXNet(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, num_segments=8, alphabet_size=4):
        super().__init__()
        self.sax_layer = DifferentiableSAX(num_segments=num_segments, alphabet_size=alphabet_size)
        sax_out_dim = num_segments * alphabet_size

        self.mlp = nn.Sequential(
            nn.Linear(sax_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        sax_features = self.sax_layer(x)
        sax_features = sax_features.view(x.shape[0], -1)
        return self.mlp(sax_features)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
