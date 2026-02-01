import torch
import torch.nn as nn
import math

class FractionalFilterbank(nn.Module):
    def __init__(self, in_channels, out_channels_per_in, init_alpha=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_per_in = out_channels_per_in
        # Initialize alpha_raw so that sigmoid(alpha_raw)*2 is around init_alpha
        # sigmoid(x) = init_alpha / 2
        # x = logit(init_alpha / 2)
        # Randomized initialization around init_alpha
        val = math.log((init_alpha / 2.0) / (1.0 - init_alpha / 2.0))
        self.alpha_raw = nn.Parameter(torch.full((in_channels, out_channels_per_in), val) + torch.randn(in_channels, out_channels_per_in) * 0.5)
        self.gain = nn.Parameter(torch.ones(in_channels, out_channels_per_in))
        self.bias = nn.Parameter(torch.zeros(in_channels * out_channels_per_in))

    def forward(self, x):
        # x shape: [batch, in_channels, length]
        batch_size, in_channels, n = x.shape

        # Use padding to avoid circular convolution artifacts
        n_pad = n * 2
        x_fft = torch.fft.rfft(x, n=n_pad) # [batch, in_channels, n_pad//2 + 1]

        freqs = torch.fft.rfftfreq(n_pad, device=x.device) # [n_pad//2 + 1]

        alpha = torch.sigmoid(self.alpha_raw) * 2.0 # [in_channels, out_channels_per_in]

        safe_freqs = 2 * math.pi * freqs

        # alpha: [C, K, 1]
        # safe_freqs: [1, 1, F]
        alpha_exp = alpha.unsqueeze(-1)
        freqs_exp = safe_freqs.view(1, 1, -1)

        # We use a small epsilon for pow if alpha can be very small
        mag = torch.pow(freqs_exp + 1e-12, alpha_exp)
        # For fractional derivative, we typically want to zero out the DC component
        # But let's allow it to be learned if alpha is close to 0
        # Actually, let's keep it zero for now but add padding.
        mag = mag * (freqs_exp > 0).float()

        phase = alpha_exp * (math.pi / 2)

        multiplier = torch.polar(mag, phase)

        # x_fft: [B, C, 1, F]
        # multiplier: [1, C, K, F]
        output_fft = x_fft.unsqueeze(2) * multiplier.unsqueeze(0)
        output_fft = output_fft * self.gain.view(1, in_channels, self.out_channels_per_in, 1)

        # [B, C, K, F] -> [B, C, K, L_pad]
        y = torch.fft.irfft(output_fft, n=n_pad)
        # Crop to original length
        y = y[:, :, :, :n]
        y = y.reshape(batch_size, in_channels * self.out_channels_per_in, n)
        y = y + self.bias.view(1, -1, 1)
        return y

class LFDFModel(nn.Module):
    def __init__(self, input_dim=40, num_filters=32, hidden_dim=256, output_dim=10):
        super().__init__()
        self.lfdf = FractionalFilterbank(1, num_filters)
        self.bn = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(num_filters * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 1, 40]
        x = self.lfdf(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class ConvBaseline(nn.Module):
    def __init__(self, input_dim=40, num_filters=32, kernel_size=5, hidden_dim=256, output_dim=10):
        super().__init__()
        self.conv = nn.Conv1d(1, num_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(num_filters * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x
