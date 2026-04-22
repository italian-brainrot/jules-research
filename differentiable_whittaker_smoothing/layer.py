import torch
import torch.nn as nn
import torch.linalg

def get_difference_matrix(n, d=2):
    """
    Returns a d-th order difference matrix of shape (n-d, n)
    """
    if d == 0:
        return torch.eye(n)

    # Start with identity
    D = torch.eye(n)
    for _ in range(d):
        # D_new = D[1:] - D[:-1]
        D = D[1:] - D[:-1]
    return D

class WhittakerSmoothing(nn.Module):
    def __init__(self, n_features, order=2, initial_lambda=1.0, learnable_lambda=True, per_channel=False, channels=1):
        super().__init__()
        self.n_features = n_features
        self.order = order

        # Precompute D^T * D
        D = get_difference_matrix(n_features, d=order)
        DTD = D.t() @ D
        self.register_buffer("DTD", DTD)
        self.register_buffer("I", torch.eye(n_features))

        if per_channel:
            self.log_lambda = nn.Parameter(torch.full((channels, 1, 1), torch.log(torch.as_tensor(float(initial_lambda)))))
        else:
            self.log_lambda = nn.Parameter(torch.log(torch.as_tensor(float(initial_lambda))))

        self.learnable_lambda = learnable_lambda
        if not learnable_lambda:
            self.log_lambda.requires_grad = False

    def forward(self, x):
        """
        x: (batch_size, n_features) or (batch_size, channels, n_features)
        """
        original_shape = x.shape
        if x.dim() == 2:
            # (batch, n)
            x = x.unsqueeze(1) # (batch, 1, n)

        # x is (batch, channels, n)
        batch_size, channels, n = x.shape

        # A = I + lambda * DTD
        # lambda must be positive, so we use exp(log_lambda)
        lmbda = torch.exp(self.log_lambda)

        # self.DTD is (n, n)
        # lmbda is (1) or (channels, 1, 1)
        A = self.I + lmbda * self.DTD # (channels, n, n) or (n, n)

        if A.dim() == 2:
            # Broadcase A to (batch, channels, n, n)
            A = A.expand(batch_size, channels, n, n)
        else:
            # A is (channels, n, n)
            A = A.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # x is (batch, channels, n) -> need (batch, channels, n, 1) for solve
        rhs = x.unsqueeze(-1)

        # solve A z = y
        z = torch.linalg.solve(A, rhs)

        # z is (batch, channels, n, 1)
        z = z.squeeze(-1)

        if len(original_shape) == 2:
            z = z.squeeze(1)

        return z

class WhittakerMultiScale(nn.Module):
    """
    Applies Whittaker smoothing at multiple scales and concatenates the results.
    """
    def __init__(self, n_features, orders=[2], lambdas=[1.0, 10.0, 100.0], learnable=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in orders:
            for l in lambdas:
                self.layers.append(WhittakerSmoothing(n_features, order=d, initial_lambda=l, learnable_lambda=learnable))

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            outputs.append(layer(x))
        return torch.cat(outputs, dim=-1)
