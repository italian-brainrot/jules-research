import torch
from log_linear_interaction_network.model import LogLinearInteractionLayer

def test_log_linear_layer():
    batch_size = 5
    in_features = 4
    out_features = 3
    epsilon = 1e-5

    layer = LogLinearInteractionLayer(in_features, out_features, epsilon=epsilon)
    x = torch.randn(batch_size, in_features)

    # Manual calculation
    with torch.no_grad():
        # Additive path
        out_add_expected = F_linear(x, layer.linear_add.weight, layer.linear_add.bias)

        # Multiplicative path
        x_log = torch.log(torch.abs(x) + epsilon)
        x_mult_latent = F_linear(x_log, layer.linear_log.weight, layer.linear_log.bias)
        out_mult_expected = F_linear(torch.exp(x_mult_latent), layer.linear_mult.weight, layer.linear_mult.bias)

        expected = out_add_expected + out_mult_expected

    out = layer(x)

    diff = torch.abs(out - expected).max().item()
    print(f"Max difference: {diff}")
    assert diff < 1e-5, f"Difference too large: {diff}"
    print("Logic verification passed!")

def F_linear(input, weight, bias):
    return torch.matmul(input, weight.t()) + bias

if __name__ == "__main__":
    test_log_linear_layer()
