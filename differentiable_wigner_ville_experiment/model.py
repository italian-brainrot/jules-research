import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

def hilbert(x):
    # x: (batch, N)
    N = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)
    # Create h filter
    # h = [1, 2, 2, ..., 2, 1, 0, 0, ..., 0] if N is even
    # h = [1, 2, 2, ..., 2, 0, 0, ..., 0] if N is odd
    h = torch.zeros(N, device=x.device)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    # h is applied in the frequency domain
    # Use broadcasting for batch dimension
    return torch.fft.ifft(X * h.to(X.dtype), dim=-1)

class WignerVilleLayer(nn.Module):
    def __init__(self, n_fft=32, window_size=15):
        super().__init__()
        self.n_fft = n_fft
        self.window_size = window_size
        # window_size should be odd for symmetric window around lag 0
        if self.window_size % 2 == 0:
            self.window_size += 1
        self.register_buffer('window', torch.hamming_window(self.window_size))

    def forward(self, x):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.shape

        # 1. Compute analytic signal
        z = hilbert(x) # (batch, seq_len) complex

        # 2. Compute Instantaneous Autocorrelation (IA)
        # R[n, m] = z[n+m] * conj(z[n-m]) for m in [-M, M]
        M = (self.window_size - 1) // 2
        z_padded = F.pad(z, (M, M), mode='constant', value=0)

        # We want to extract z[n+m] and z[n-m]
        # For a fixed n, we want z[n+M...n-M] and z[n-M...n+M]
        # Use unfold to get windows of length 2*M + 1
        # z_padded has length seq_len + 2*M
        # unfolding with size 2*M + 1 and step 1 gives seq_len windows
        z_unfolded = z_padded.unfold(1, self.window_size, 1) # (batch, seq_len, window_size)

        # z_unfolded[:, n, :] is [z[n], z[n+1], ..., z[n+2M]]
        # Wait, the indices should be centered around n.
        # z_padded: [z_pad_0...z_pad_{M-1}, z_0, ..., z_{seq_len-1}, z_pad_0...z_pad_{M-1}]
        # unfolded[0] is z_padded[0:2M+1] = [z_pad_0...z_pad_{M-1}, z_0, z_1, ..., z_M]
        # So for time n, the window is z_padded[n : n + 2M + 1]
        # The center element is z_padded[n + M], which is z_n.
        # So z_unfolded[:, n, m] corresponds to z[n + m - M]

        # We need z[n+m] * conj(z[n-m]) for m in [-M, M]
        # Let m' = m + M, so m' in [0, 2M]
        # R[n, m'] = z[n + (m'-M)] * conj(z[n - (m'-M)])
        # m' - M ranges from -M to M.
        # When m' = 0, m'-M = -M. R[n, 0] = z[n-M] * conj(z[n+M])
        # When m' = M, m'-M = 0. R[n, M] = z[n] * conj(z[n]) = |z[n]|^2
        # When m' = 2M, m'-M = M. R[n, 2M] = z[n+M] * conj(z[n-M])

        # z_unfolded contains [z[n-M], ..., z[n], ..., z[n+M]]
        # Let's call this Z_n
        # Z_n_reversed is [z[n+M], ..., z[n], ..., z[n-M]]

        z_unfolded_reversed_conj = z_unfolded.conj().flip(-1)

        R = z_unfolded * z_unfolded_reversed_conj * self.window # (batch, seq_len, window_size)

        # 3. FFT over the lag dimension
        W = torch.fft.fft(R, n=self.n_fft, dim=-1)

        # WVD is real. We use .real to get rid of numerical noise in imag part
        return W.real

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10):
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

class WVMLP(nn.Module):
    def __init__(self, input_len=40, n_fft=32, window_size=15, hidden_dim=128, output_dim=10):
        super().__init__()
        self.wv = WignerVilleLayer(n_fft=n_fft, window_size=window_size)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_len * n_fft, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        x_wv = self.wv(x)
        return self.net(x_wv)

class WVAugmentedMLP(nn.Module):
    def __init__(self, input_len=40, n_fft=32, window_size=15, hidden_dim=128, output_dim=10):
        super().__init__()
        self.wv = WignerVilleLayer(n_fft=n_fft, window_size=window_size)
        self.net = nn.Sequential(
            nn.Linear(input_len + input_len * n_fft, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        x_wv = self.wv(x)
        x_combined = torch.cat([x, x_wv.flatten(1)], dim=1)
        return self.net(x_combined)
