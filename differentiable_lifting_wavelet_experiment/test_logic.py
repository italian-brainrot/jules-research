import torch
from model import LiftingLayer, LLWNet

def test_lifting_invertibility():
    layer = LiftingLayer(kernel_size=3)
    x = torch.randn(5, 1, 40)
    a, d = layer(x)
    x_rec = layer.inverse(a, d)

    diff = torch.max(torch.abs(x - x_rec))
    print(f"Lifting layer reconstruction error: {diff.item()}")
    assert diff < 1e-5

def test_gradients():
    model = LLWNet(input_dim=40, levels=2, kernel_size=3)
    x = torch.randn(5, 40, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    print("Gradients to input x:", x.grad is not None)
    for i, layer in enumerate(model.layers):
        print(f"Gradients to P of layer {i}:", layer.P.weight.grad is not None)
        print(f"Gradients to U of layer {i}:", layer.U.weight.grad is not None)
        print(f"Gradients to s of layer {i}:", layer.s.grad is not None)

    assert x.grad is not None
    for layer in model.layers:
        assert layer.P.weight.grad is not None
        assert layer.U.weight.grad is not None
        assert layer.s.grad is not None

if __name__ == "__main__":
    test_lifting_invertibility()
    test_gradients()
