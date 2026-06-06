import torch
import torch.nn as nn
import torch.nn.functional as F

class DICALayer(nn.Module):
    def __init__(self, num_features, num_components=None, iterations=10, eps=1e-5):
        super(DICALayer, self).__init__()
        self.num_features = num_features
        self.num_components = num_components or num_features
        self.iterations = iterations
        self.eps = eps

        # Initialize W as identity for deterministic behavior
        self.register_buffer('W_init', torch.eye(self.num_components, self.num_features))

    def _symmetric_decorrelation(self, W):
        u, s, vh = torch.linalg.svd(W, full_matrices=False)
        return u @ vh

    def forward(self, x):
        # x: (batch, features)
        B, F = x.shape
        if B <= 1:
             return torch.zeros(B, self.num_components, device=x.device)

        # 1. Centering
        mean = x.mean(dim=0, keepdim=True)
        x_c = x - mean

        # 2. Whitening
        cov = (x_c.t() @ x_c) / (B - 1)
        cov = cov + torch.eye(F, device=x.device) * self.eps

        L, V = torch.linalg.eigh(cov)
        L = torch.clamp(L, min=self.eps)
        whitening_matrix = V @ torch.diag(L.pow(-0.5)) @ V.t()

        x_w = x_c @ whitening_matrix.t()

        # 3. FastICA unrolled iterations
        W = self.W_init.to(x.device)

        for _ in range(self.iterations):
            u = x_w @ W.t()
            g_u = torch.tanh(u)
            dg_u = 1.0 - g_u.pow(2)

            term1 = (g_u.t() @ x_w) / B
            term2 = dg_u.mean(dim=0).unsqueeze(1) * W

            W = term1 - term2
            W = self._symmetric_decorrelation(W)

        # 4. Extract and Stabilize components
        s = x_w @ W.t() # (B, num_components)

        # Sign normalization: Ensure positive skewness
        # (Actually, ICA components are often zero-skew if symmetric,
        # but we can use some convention like max absolute value or skewness)
        skew = (s**3).mean(dim=0)
        signs = torch.sign(skew + 1e-9)
        s = s * signs.unsqueeze(0)

        # Sorting: Sort by kurtosis (measure of non-Gaussianity)
        # kurtosis = E[x^4] / (E[x^2]^2) - 3. Since x_w is whitened, E[x^2] approx 1.
        kurt = (s**4).mean(dim=0) - 3.0
        _, idx = torch.sort(kurt.abs(), descending=True)
        s = s[:, idx]

        return s

class PCALayer(nn.Module):
    def __init__(self, num_features, num_components=None, eps=1e-5):
        super(PCALayer, self).__init__()
        self.num_features = num_features
        self.num_components = num_components or num_features
        self.eps = eps

    def forward(self, x):
        B, F = x.shape
        if B <= 1:
             return torch.zeros(B, self.num_components, device=x.device)

        mean = x.mean(dim=0, keepdim=True)
        x_c = x - mean

        cov = (x_c.t() @ x_c) / (B - 1)
        cov = cov + torch.eye(F, device=x.device) * self.eps

        L, V = torch.linalg.eigh(cov)
        idx = torch.argsort(L, descending=True)
        L = L[idx]
        V = V[:, idx]

        V = V[:, :self.num_components]
        L = L[:self.num_components]

        # Whitened PCA
        s = x_c @ V @ torch.diag(L.clamp(min=self.eps).pow(-0.5))

        # Sign normalization for PCA: Ensure max abs value is positive
        max_abs_idx = torch.argmax(torch.abs(s), dim=0)
        signs = torch.sign(s[max_abs_idx, torch.arange(s.shape[1])])
        s = s * signs.unsqueeze(0)

        return s

class ICAClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mode='baseline', ica_iters=10):
        super(ICAClassifier, self).__init__()
        self.mode = mode
        if mode == 'ica':
            self.pre = DICALayer(input_dim, iterations=ica_iters)
            feat_dim = input_dim
        elif mode == 'pca':
            self.pre = PCALayer(input_dim)
            feat_dim = input_dim
        elif mode == 'ica_aug':
            self.pre = DICALayer(input_dim, iterations=ica_iters)
            feat_dim = input_dim * 2
        elif mode == 'pca_aug':
            self.pre = PCALayer(input_dim)
            feat_dim = input_dim * 2
        else:
            self.pre = nn.Identity()
            feat_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if self.mode in ['ica', 'pca']:
            x = self.pre(x)
        elif self.mode in ['ica_aug', 'pca_aug']:
            x_feat = self.pre(x)
            x = torch.cat([x, x_feat], dim=1)
        return self.mlp(x)
