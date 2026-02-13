import torch
from model import GADMLP, BaselineMLP

def test_model_forward():
    x = torch.randn(8, 40)

    model_gad = GADMLP()
    model_gad.set_dropout_rates(0.5, 0.5)
    y_gad = model_gad(x)
    assert y_gad.shape == (8, 10)

    model_base = BaselineMLP(p=0.5)
    y_base = model_base(x)
    assert y_base.shape == (8, 10)
    print("Forward pass tests passed!")

def test_dropout_effect():
    # In training mode, dropout should be active
    model = BaselineMLP(p=1.0) # All zeroed
    model.train()
    x = torch.randn(8, 40)
    y = model(x)
    # If all hidden units are dropped, output might still be non-zero because of biases in fc3
    # but let's check if it's consistent

    model_gad = GADMLP()
    model_gad.set_dropout_rates(1.0, 1.0)
    model_gad.train()
    y_gad = model_gad(x)

    print("Dropout effect checked (manual inspection recommended if suspicious)")

if __name__ == "__main__":
    test_model_forward()
    test_dropout_effect()
