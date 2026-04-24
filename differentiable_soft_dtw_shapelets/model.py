import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_dtw_core_vectorized(D, gamma):
    # D: (B, N, M)
    B, N, M = D.shape
    device = D.device
    dtype = D.dtype

    # Initialize DP table
    # We use a slightly different approach for vectorization
    # R: (B, N + 1, M + 1)
    R = torch.full((B, N + 1, M + 1), 1e10, device=device, dtype=dtype)
    R[:, 0, 0] = 0.0

    # To vectorize, we can't easily use JIT on the whole thing if we want complex indexing.
    # But let's see.
    for k in range(2, N + M + 1):
        # i + j = k, 1 <= i <= N, 1 <= j <= M
        i_min = max(1, k - M)
        i_max = min(k - 1, N)

        i = torch.arange(i_min, i_max + 1, device=device)
        j = k - i

        # We want to compute R[:, i, j]
        # Previous values:
        # a = R[:, i-1, j-1]
        # b = R[:, i-1, j]
        # c = R[:, i, j-1]

        a = R[:, i - 1, j - 1]
        b = R[:, i - 1, j]
        c = R[:, i, j - 1]

        # m: (B, num_elements_in_diag, 3)
        m = torch.stack([a, b, c], dim=-1)
        soft_min = -gamma * torch.logsumexp(-m / gamma, dim=-1)

        # Update diagonal
        # This is the tricky part in PyTorch for vectorization.
        # We can use scatter or advanced indexing.
        R[:, i, j] = D[:, i - 1, j - 1] + soft_min

    return R[:, N, M]

# Let's use a more efficient version of Soft-DTW if possible.
# Actually, for N, M = 10, the overhead is mostly the PyTorch kernel launches.
# If I can use a single JIT function for the loop, it's better.

@torch.jit.script
def soft_dtw_jit(D, gamma):
    B, N, M = D.shape
    device = D.device
    dtype = D.dtype
    R = torch.full((B, N + 1, M + 1), 1e10, device=device, dtype=dtype)
    R[:, 0, 0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            a = R[:, i - 1, j - 1]
            b = R[:, i - 1, j]
            c = R[:, i, j - 1]
            # Manual logsumexp for 3 elements to avoid stack/logsumexp overhead
            # softmin(a, b, c) = -gamma * log(exp(-a/g) + exp(-b/g) + exp(-c/g))

            # For numerical stability:
            # -gamma * (max_val + log(exp((-a/g)-max_val) + ...))
            # where max_val = max(-a/g, -b/g, -c/g)

            ag = -a / gamma
            bg = -b / gamma
            cg = -c / gamma

            max_val = torch.max(torch.max(ag, bg), cg)
            soft_min = -gamma * (max_val + torch.log(torch.exp(ag - max_val) + torch.exp(bg - max_val) + torch.exp(cg - max_val)))

            R[:, i, j] = D[:, i - 1, j - 1] + soft_min
    return R[:, N, M]

class SoftDTWShapeletLayer(nn.Module):
    def __init__(self, in_channels, num_shapelets, shapelet_length, gamma=1.0, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_shapelets = num_shapelets
        self.shapelet_length = shapelet_length
        self.stride = stride
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

        self.shapelets = nn.Parameter(torch.randn(num_shapelets, in_channels, shapelet_length))

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        K = self.shapelet_length

        x_unfolded = x.unfold(2, K, self.stride)
        num_windows = x_unfolded.shape[2]

        xp = x_unfolded.permute(0, 2, 3, 1).reshape(-1, K, C)
        sq = self.shapelets.permute(0, 2, 1)

        NW, NS = xp.shape[0], sq.shape[0]

        xp_norm = torch.sum(xp**2, dim=2)
        sq_norm = torch.sum(sq**2, dim=2)
        cross = torch.einsum('wic,sjc->wsij', xp, sq)
        dist_mat = xp_norm.view(NW, 1, K, 1) + sq_norm.view(1, NS, 1, K) - 2 * cross
        dist_mat = dist_mat.reshape(NW * NS, K, K)

        g = torch.clamp(self.gamma, min=1e-3)
        res = soft_dtw_jit(dist_mat, g)

        res = res.view(B, num_windows, NS)
        window_temp = 1.0
        soft_min_weights = F.softmax(-res / window_temp, dim=1)
        pooled_distances = torch.sum(res * soft_min_weights, dim=1)

        return pooled_distances

class EuclideanShapeletLayer(nn.Module):
    def __init__(self, in_channels, num_shapelets, shapelet_length):
        super().__init__()
        self.in_channels = in_channels
        self.num_shapelets = num_shapelets
        self.shapelet_length = shapelet_length
        self.shapelets = nn.Parameter(torch.randn(num_shapelets, in_channels, shapelet_length))

    def forward(self, x):
        B, C, L = x.shape
        K = self.shapelet_length
        x_unfolded = x.unfold(2, K, 1)
        x_unfolded = x_unfolded.permute(0, 2, 1, 3) # (B, num_windows, C, K)

        s = self.shapelets.view(1, 1, self.num_shapelets, self.in_channels, K)
        x_unfolded = x_unfolded.unsqueeze(2) # (B, num_windows, 1, C, K)

        dist = torch.sum((x_unfolded - s)**2, dim=(3, 4))

        window_temp = 1.0
        soft_min_weights = F.softmax(-dist / window_temp, dim=1)
        pooled_distances = torch.sum(dist * soft_min_weights, dim=1)

        return pooled_distances

class ShapeletNetwork(nn.Module):
    def __init__(self, in_channels, num_shapelets, shapelet_length, num_classes, layer_type='euclidean', gamma=1.0, stride=1):
        super().__init__()
        if layer_type == 'euclidean':
            self.shapelet_layer = EuclideanShapeletLayer(in_channels, num_shapelets, shapelet_length)
        else:
            self.shapelet_layer = SoftDTWShapeletLayer(in_channels, num_shapelets, shapelet_length, gamma, stride=stride)

        self.classifier = nn.Sequential(
            nn.Linear(num_shapelets, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.shapelet_layer(x)
        logits = self.classifier(features)
        return logits

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
