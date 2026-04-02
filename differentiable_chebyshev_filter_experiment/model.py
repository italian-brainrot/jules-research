import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DifferentiableChebyshevFilterLayer(nn.Module):
    """
    A differentiable Chebyshev Type I filter layer for 1D signals.
    Parameters:
        in_channels (int): Number of input channels.
        order (int): Filter order (n).
        initial_cutoff (float): Initial cutoff frequency (0 to 1, normalized).
        initial_ripple (float): Initial ripple (epsilon).
    """
    def __init__(self, in_channels, order=2, initial_cutoff=0.5, initial_ripple=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.order = order

        # Learnable parameters: cutoff frequency and ripple
        # Use sigmoid to keep cutoff between 0 and 1
        self.raw_cutoff = nn.Parameter(torch.full((in_channels,), math.log(initial_cutoff / (1 - initial_cutoff))))
        # Use softplus to keep ripple positive
        self.raw_ripple = nn.Parameter(torch.full((in_channels,), math.log(math.exp(initial_ripple) - 1)))

    def forward(self, x):
        """
        x shape: (batch, in_channels, length)
        """
        batch_size, in_channels, length = x.shape
        cutoff = torch.sigmoid(self.raw_cutoff) * 0.45 + 0.01 # Keep it in a safe range [0.01, 0.46]
        eps = F.softplus(self.raw_ripple) + 1e-6

        # Design Chebyshev Type I filter in s-domain and use bilinear transform
        # For simplicity and stability, we implement second-order sections (SOS) if order > 2
        # But for the experiment, let's start with a fixed order (e.g., 2) for clarity.
        # Order 2 Chebyshev Type I:
        # H(s) = 1 / [s^2 + s * 2 * sinh(as) * sin(theta) + sinh^2(as) + cos^2(theta)]
        # where as = 1/n * asinh(1/eps), theta = (2k-1)*pi/(2n)

        # We'll implement a 2nd order filter directly.
        # k=1: theta = pi / 4
        n = self.order
        theta = math.pi / (2 * n)
        phi = (1 / n) * torch.asinh(1 / eps)

        # Poles in snd-domain (normalized cutoff=1)
        # s_p = -sinh(phi)sin((2k-1)pi/2n) + i cosh(phi)cos((2k-1)pi/2n)
        # For n=2, k=1, 2:
        # s_1,2 = -sinh(phi)sin(pi/4) +/- i cosh(phi)cos(pi/4)

        s_real = -torch.sinh(phi) * math.sin(math.pi / 4)
        s_imag = torch.cosh(phi) * math.cos(math.pi / 4)

        # H(s) = Gain / [(s - s_1)(s - s_2)] = Gain / [s^2 - 2*s_real*s + (s_real^2 + s_imag^2)]
        # Normalize so H(0) = 1 (for even n, Chebyshev I has H(0) = 1/sqrt(1+eps^2) actually,
        # but let's use standard normalization or adjust gain).
        # Actually for Chebyshev I, at w=0, |H| = 1 if n is odd, |H| = 1/sqrt(1+eps^2) if n is even.

        omega_c = 2 * math.pi * cutoff # Approximate analog cutoff

        # Denominator in s-domain: s^2 + a1*s + a0
        a1 = -2 * s_real * omega_c
        a0 = (s_real**2 + s_imag**2) * (omega_c**2)

        # Bilinear transform: s = 2 * (1 - z^-1) / (1 + z^-1)
        # H(z) = Gain * (1 + z^-1)^2 / [ (2(1-z^-1)/(1+z^-1))^2 + a1*(2(1-z^-1)/(1+z^-1)) + a0 ]
        # H(z) = Gain * (1 + z^-1)^2 / [ 4(1 - 2z^-1 + z^-2) + 2*a1(1 - z^-2) + a0(1 + 2z^-1 + z^-2) ]
        # H(z) = Gain * (1 + 2z^-1 + z^-2) / [ (4 + 2*a1 + a0) + (-8 + 2*a0)z^-1 + (4 - 2*a1 + a0)z^-2 ]

        b0 = 1.0
        b1 = 2.0
        b2 = 1.0

        v0 = 4 + 2*a1 + a0
        v1 = -8 + 2*a0
        v2 = 4 - 2*a1 + a0

        # Gain to make it match Chebyshev response
        # At s=0, H(s) = Gain / a0.
        # We want H(0) = 1 (for simplicity, or 1/sqrt(1+eps^2))
        gain = a0
        if n % 2 == 0:
            gain = gain / torch.sqrt(1 + eps**2)

        # Coefficients for each channel
        # Shape: (in_channels, 3)
        # (batch, in_channels)
        B0 = (gain * b0 / v0).unsqueeze(0)
        B1 = (gain * b1 / v0).unsqueeze(0)
        B2 = (gain * b2 / v0).unsqueeze(0)
        A1 = (v1 / v0).unsqueeze(0)
        A2 = (v2 / v0).unsqueeze(0)

        # Apply filter using difference equation:
        # y[t] = b0*x[t] + b1*x[t-1] + b2*x[t-2] - a1*y[t-1] - a2*y[t-2]

        y_list = []

        y_prev1 = torch.zeros(batch_size, in_channels, device=x.device)
        y_prev2 = torch.zeros(batch_size, in_channels, device=x.device)
        x_prev1 = torch.zeros(batch_size, in_channels, device=x.device)
        x_prev2 = torch.zeros(batch_size, in_channels, device=x.device)

        for t in range(length):
            x_t = x[:, :, t]
            y_t = (B0 * x_t + B1 * x_prev1 + B2 * x_prev2 -
                   A1 * y_prev1 - A2 * y_prev2)
            y_list.append(y_t)

            x_prev2 = x_prev1
            x_prev1 = x_t
            y_prev2 = y_prev1
            y_prev1 = y_t

        return torch.stack(y_list, dim=2)

class ChebyshevConvMLP(nn.Module):
    def __init__(self, input_dim=40, in_channels=1, hidden_dim=256, output_dim=10, order=2):
        super().__init__()
        self.cheby = DifferentiableChebyshevFilterLayer(in_channels, order=order)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, 40)
        x = x.unsqueeze(1) # (batch, 1, 40)
        x = self.cheby(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

class BaselineConvMLP(nn.Module):
    def __init__(self, input_dim=40, in_channels=1, hidden_dim=256, output_dim=10):
        super().__init__()
        # Use a simple 1D convolution as a baseline for comparison
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # (batch, 1, 40)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

if __name__ == "__main__":
    layer = DifferentiableChebyshevFilterLayer(1)
    x = torch.randn(2, 1, 40)
    y = layer(x)
    print(y.shape)
