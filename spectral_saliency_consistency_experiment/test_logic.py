import torch
import torch.nn as nn
from spectral_saliency_consistency_experiment.model import get_model
from spectral_saliency_consistency_experiment.train import compute_ssc_loss

def test_ssc_loss_flow():
    device = torch.device('cpu')
    batch_size = 8
    input_dim = 40
    num_classes = 10

    model = get_model(input_dim, 64, num_classes).to(device)
    x = torch.randn(batch_size, input_dim, requires_grad=True).to(device)
    y = torch.randint(0, num_classes, (batch_size,)).to(device)

    lambda_ssc = 0.5

    # Check if we can compute the loss
    loss = compute_ssc_loss(model, x, y, lambda_ssc)
    print(f"SSC Loss: {loss.item()}")

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

    # Check if gradients flow back to model parameters
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.norm().item()}")
        else:
            print(f"NO gradient for {name}")
            # Biases might have zero gradient in some cases, but weights should have some
            if "weight" in name:
                assert param.grad is not None and param.grad.norm().item() > 0

    print("Test passed!")

if __name__ == "__main__":
    test_ssc_loss_flow()
