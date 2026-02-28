import torch
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call
from model import MLP

def test_igsr_logic():
    input_dim = 10
    batch_size = 4
    hidden_dim = 16
    output_dim = 5

    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    x = torch.randn(batch_size, input_dim, requires_grad=True)
    y = torch.randint(0, output_dim, (batch_size,))

    def compute_loss_single(params, buffers, x_single, y_single):
        # x_single: (input_dim,)
        # y_single: ()
        logits = functional_call(model, (params, buffers), (x_single.unsqueeze(0),))
        loss = F.cross_entropy(logits, y_single.unsqueeze(0))
        return loss

    # Compute per-sample input gradients
    # in_dims: params=None, buffers=None, x=0, y=0
    grad_fn = vmap(grad(compute_loss_single, argnums=2), in_dims=(None, None, 0, 0))
    input_grads = grad_fn(params, buffers, x, y)

    print(f"Input grads shape: {input_grads.shape}")
    assert input_grads.shape == (batch_size, input_dim)

    # IGSR penalty: mean over batch and spatial diff
    # input_grads: (batch_size, input_dim)
    diffs = input_grads[:, 1:] - input_grads[:, :-1]
    igsr_penalty = torch.mean(diffs**2)

    print(f"IGSR penalty: {igsr_penalty.item()}")
    assert igsr_penalty.item() >= 0

    # Check if we can backprop through it
    loss_total = igsr_penalty
    loss_total.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Param {name} grad norm: {param.grad.norm().item()}")
        else:
            print(f"Param {name} grad is None")

if __name__ == "__main__":
    test_igsr_logic()
