import torch
from model import BaselineMLP, GRDMLP, NGRDMLP

def test_models():
    input_size = 40
    hidden_size = 256
    output_size = 10
    x = torch.randn(5, input_size)

    models = [
        BaselineMLP(input_size, hidden_size, output_size, p=0.2),
        GRDMLP(input_size, hidden_size, output_size),
        NGRDMLP(input_size, hidden_size, output_size)
    ]

    for model in models:
        y = model(x)
        assert y.shape == (5, output_size)
        print(f"Model {model.__class__.__name__} passed.")

if __name__ == "__main__":
    test_models()
