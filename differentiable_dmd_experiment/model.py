import torch
import torch.nn as nn
import torch.nn.functional as F

class DDMDLayer(nn.Module):
    """
    Differentiable Dynamic Mode Decomposition Layer.
    Extracts eigenvalues of the transition operator from a 1D signal using time-delay embedding.
    """
    def __init__(self, L=40, k=15):
        super().__init__()
        self.L = L
        self.k = k
        if k >= L:
            raise ValueError(f"k ({k}) must be less than L ({L})")

    def forward(self, x):
        # x: (B, L)
        B, L = x.shape
        k = self.k
        # unfold(1, k, 1) -> (B, n, k) where n = L - k + 1
        # n is the number of windows
        # k is the size of each window (embedding dimension)
        H = x.unfold(1, k, 1).transpose(1, 2) # (B, k, n)

        X = H[:, :, :-1] # (B, k, n-1)
        Y = H[:, :, 1:]  # (B, k, n-1)

        # Solve AX = Y => A = Y X^dagger
        # We solve Xt At = Yt for At
        Xt = X.transpose(-2, -1) # (B, n-1, k)
        Yt = Y.transpose(-2, -1) # (B, n-1, k)

        # Use differentiable least squares
        # solve solves AX = B for X, where A is (..., M, N), B is (..., M, K)
        # Here we want Xt At = Yt, so A=Xt, B=Yt, X=At.
        # Xt is (B, n-1, k), Yt is (B, n-1, k)
        # At will be (B, k, k)
        sol = torch.linalg.lstsq(Xt, Yt).solution
        A = sol.transpose(-2, -1) # (B, k, k)

        # Eigenvalues
        # evals: (B, k) complex
        # We use eigvals which is differentiable as long as eigenvalues are distinct.
        evals = torch.linalg.eigvals(A)

        # Sort by magnitude for stability
        mags = torch.abs(evals)
        # Also consider sorting by real part if magnitudes are similar
        # But magnitude is standard for DMD (identifying stable/unstable modes)
        _, indices = torch.sort(mags, dim=-1, descending=True)

        # gather sorted eigenvalues
        sorted_evals = torch.gather(evals, -1, indices)

        # Features: real and imaginary parts
        features = torch.cat([sorted_evals.real, sorted_evals.imag], dim=-1) # (B, 2k)

        return features

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

class DDMDNet(nn.Module):
    def __init__(self, L=40, k=15, hidden_dim=256, output_dim=10):
        super().__init__()
        self.ddmd = DDMDLayer(L=L, k=k)
        self.net = nn.Sequential(
            nn.Linear(2 * k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, L)
        feat = self.ddmd(x)
        return self.net(feat)

class DDMDAugmentedMLP(nn.Module):
    def __init__(self, L=40, k=15, hidden_dim=256, output_dim=10):
        super().__init__()
        self.ddmd = DDMDLayer(L=L, k=k)
        self.net = nn.Sequential(
            nn.Linear(L + 2 * k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, L)
        feat = self.ddmd(x)
        combined = torch.cat([x, feat], dim=1)
        return self.net(combined)

if __name__ == "__main__":
    # Test shapes
    B, L, k = 4, 40, 15
    x = torch.randn(B, L)
    layer = DDMDLayer(L=L, k=k)
    out = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}") # Should be (B, 2k) = (4, 30)

    model = DDMDAugmentedMLP(L=L, k=k)
    out_model = model(x)
    print(f"Model output shape: {out_model.shape}")
