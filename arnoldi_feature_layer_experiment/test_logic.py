import torch
import torch.nn as nn

def arnoldi_process(A, x, k, eps=1e-8):
    """
    Performs k steps of the Arnoldi process for matrix A and starting vector x.
    Returns the upper Hessenberg matrix H.
    """
    batch_size, d = x.shape
    q = x / (torch.norm(x, dim=1, keepdim=True) + eps)

    # Store Q vectors to orthogonalize against
    qs = [q]
    H = torch.zeros(batch_size, k + 1, k, device=x.device, dtype=x.dtype)

    for j in range(k):
        v = torch.matmul(qs[j].unsqueeze(1), A).squeeze(1) # (B, d)

        for i in range(j + 1):
            h_ij = torch.sum(qs[i] * v, dim=1) # (B,)
            H[:, i, j] = h_ij
            v = v - h_ij.unsqueeze(1) * qs[i]

        h_next_j = torch.norm(v, dim=1)
        H[:, j + 1, j] = h_next_j
        qs.append(v / (h_next_j.unsqueeze(1) + eps))

    return H

class ArnoldiFeatureLayer(nn.Module):
    def __init__(self, input_dim, k, num_heads=1):
        super().__init__()
        self.input_dim = input_dim
        self.k = k
        self.num_heads = num_heads
        # Learnable matrices A
        self.A = nn.Parameter(torch.randn(num_heads, input_dim, input_dim) * 0.01)

    def forward(self, x):
        # x: (B, d)
        B, d = x.shape
        all_H = []
        for i in range(self.num_heads):
            H = arnoldi_process(self.A[i], x, self.k) # (B, k+1, k)
            all_H.append(H.reshape(B, -1))

        return torch.cat(all_H, dim=1)

def test_differentiation():
    d = 10
    k = 3
    x = torch.randn(2, d, requires_grad=True)
    A = torch.randn(d, d, requires_grad=True)

    H = arnoldi_process(A, x, k)
    loss = H.sum()
    loss.backward()

    print("X grad norm:", x.grad.norm().item())
    print("A grad norm:", A.grad.norm().item())
    assert x.grad is not None
    assert A.grad is not None
    print("Differentiation successful!")

if __name__ == "__main__":
    test_differentiation()
