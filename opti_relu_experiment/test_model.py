import torch
from opti_relu_experiment.model import OptiReLU, train_optirelu

def test_optirelu_init():
    model = OptiReLU(10, 5)
    assert model.input_dim == 10
    assert model.output_dim == 5
    assert model.hidden_weights.shape[0] == 0

def test_optirelu_forward():
    model = OptiReLU(10, 5)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 5)

def test_optirelu_add_neuron():
    model = OptiReLU(10, 5)
    w = torch.randn(10)
    b = torch.randn(1)
    model.add_neuron(w, b)
    assert model.hidden_weights.shape[0] == 1
    out = model(torch.randn(8, 10))
    assert out.shape == (8, 5)

if __name__ == "__main__":
    test_optirelu_init()
    test_optirelu_forward()
    test_optirelu_add_neuron()
    print("All tests passed!")
