import torch
from model import DVGLayer, DVGMLP, DVGGNN

def test_dvg_layer():
    L = 10
    batch_size = 2
    y = torch.randn(batch_size, L, requires_grad=True)
    layer = DVGLayer(L=L, initial_scale=10.0)

    A = layer(y)

    # Check shape
    assert A.shape == (batch_size, L, L)

    # Check symmetry
    assert torch.allclose(A, A.transpose(1, 2))

    # Check diagonal is zero
    diag_sum = torch.diagonal(A, dim1=1, dim2=2).sum()
    assert torch.allclose(diag_sum, torch.tensor(0.0))

    # Check differentiability
    loss = A.sum()
    loss.backward()
    assert y.grad is not None
    assert not torch.isnan(y.grad).any()
    print("DVGLayer tests passed!")

def test_visibility_logic():
    # Simple case: concave up signal
    # y = x^2, nodes 0 and 2 should see each other if node 1 is lower than the line
    L = 3
    y = torch.tensor([[0.0, -1.0, 0.0]], requires_grad=True)
    layer = DVGLayer(L=L, initial_scale=100.0) # High scale for hard-ish decision
    A = layer(y)

    # Nodes 0 and 2 should be connected
    # i=0, j=2, k=1
    # V = y0 + (y2-y0)*(t1-t0)/(t2-t0) - y1
    # V = 0 + (0-0)*(0.5-0)/(1-0) - (-1) = 1
    # sigmoid(100 * 1) approx 1
    assert A[0, 0, 2] > 0.9

    # Concave down signal
    y = torch.tensor([[0.0, 2.0, 0.0]], requires_grad=True)
    A = layer(y)
    # V = 0 + 0 - 2 = -2
    # sigmoid(100 * -2) approx 0
    assert A[0, 0, 2] < 0.1
    print("Visibility logic tests passed!")

def test_models():
    L = 40
    batch_size = 2
    x = torch.randn(batch_size, L)

    mlp = DVGMLP(L=L)
    out_mlp = mlp(x)
    assert out_mlp.shape == (batch_size, 10)

    gnn = DVGGNN(L=L)
    out_gnn = gnn(x)
    assert out_gnn.shape == (batch_size, 10)

    print("Model tests passed!")

if __name__ == "__main__":
    test_dvg_layer()
    test_visibility_logic()
    test_models()
