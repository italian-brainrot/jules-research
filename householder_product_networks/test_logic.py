import torch
import torch.nn as nn
import torch.nn.functional as F

class HouseholderLinear(nn.Module):
    def __init__(self, features, num_reflectors, bias=True):
        super().__init__()
        self.features = features
        self.num_reflectors = num_reflectors

        # v must be unit norm for Householder reflector H = I - 2vv^T
        self.v = nn.Parameter(torch.randn(num_reflectors, features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: (batch, features)
        res = x
        for i in range(self.num_reflectors):
            v = self.v[i]
            v = v / (torch.norm(v) + 1e-8)
            # Hv = (I - 2vv^T)x = x - 2v(v^Tx)
            dot = torch.matmul(res, v) # (batch,)
            res = res - 2.0 * torch.outer(dot, v)

        if self.bias is not None:
            res = res + self.bias
        return res

def test_orthogonality():
    features = 10
    num_reflectors = 5
    layer = HouseholderLinear(features, num_reflectors, bias=False)

    # Test if it preserves norm
    x = torch.randn(3, features)
    norm_x = torch.norm(x, dim=1)
    y = layer(x)
    norm_y = torch.norm(y, dim=1)

    print(f"Norm x: {norm_x}")
    print(f"Norm y: {norm_y}")
    assert torch.allclose(norm_x, norm_y, atol=1e-5)

    # Test if it's an orthogonal transformation by checking y1^T y2 = x1^T x2
    x1 = torch.randn(features)
    x2 = torch.randn(features)
    y1 = layer(x1.unsqueeze(0)).squeeze(0)
    y2 = layer(x2.unsqueeze(0)).squeeze(0)

    dot_x = torch.dot(x1, x2)
    dot_y = torch.dot(y1, y2)
    print(f"Dot x: {dot_x.item():.6f}, Dot y: {dot_y.item():.6f}")
    assert torch.allclose(dot_x, dot_y, atol=1e-5)

def test_gradients():
    features = 10
    num_reflectors = 5
    layer = HouseholderLinear(features, num_reflectors)
    x = torch.randn(3, features, requires_grad=True)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert layer.v.grad is not None
    print("Gradients computed successfully")

if __name__ == "__main__":
    test_orthogonality()
    test_gradients()
