import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def compute_jacobian_penalties(model, x, compute_cjor=True, compute_jfnr=True):
    """
    Computes Jacobian-based penalties.
    JFNR: Jacobian Frobenius Norm Regularization (sum of squares of all Jacobian elements)
    CJOR: Class-wise Jacobian Orthogonality Regularization (squared cosine similarity between class gradients)
    """
    params = dict(model.named_parameters())

    def functional_model(params, x):
        # x is (D,)
        # Use functional_call to evaluate the model with given params
        return torch.func.functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)

    # Compute per-sample Jacobian w.r.t input x: (B, C, D)
    # jacrev(functional_model, argnums=1) computes d(output)/d(x)
    # vmap allows us to compute this efficiently over the batch
    batch_jac = torch.vmap(torch.func.jacrev(functional_model, argnums=1), in_dims=(None, 0))(params, x)

    jfnr = torch.tensor(0.0, device=x.device)
    if compute_jfnr:
        # JFNR: Mean squared Frobenius norm of Jacobian per sample
        jfnr = (batch_jac**2).sum(dim=(1, 2)).mean()

    cjor = torch.tensor(0.0, device=x.device)
    if compute_cjor:
        # CJOR: Mean squared cosine similarity between gradients of different classes
        # batch_jac: (B, C, D)
        # Inner products A = JJ^T: (B, C, C)
        A = torch.bmm(batch_jac, batch_jac.transpose(1, 2))

        # Norms squared of gradients: (B, C)
        diag_A = torch.diagonal(A, dim1=1, dim2=2)
        norms = torch.sqrt(diag_A + 1e-8)

        # Cosine similarity matrix: (B, C, C)
        # S_ij = A_ij / (norm_i * norm_j)
        # denominator: (B, C, 1) * (B, 1, C) -> (B, C, C)
        denom = norms.unsqueeze(2) * norms.unsqueeze(1) + 1e-8
        S = A / denom

        # Mask off-diagonal elements
        C = S.shape[1]
        mask = ~torch.eye(C, dtype=torch.bool, device=x.device)
        mask = mask.unsqueeze(0).expand(S.shape[0], -1, -1)

        # We want S_ij to be close to 0 for i != j
        # S[mask] selects all off-diagonal elements across the batch
        cjor = (S[mask]**2).mean()

    return jfnr, cjor
