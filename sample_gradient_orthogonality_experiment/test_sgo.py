import torch
from sgo_lib import MLP, get_sgo_penalty

def test_sgo():
    model = MLP(input_size=10, hidden_size=20, output_size=5)
    params = dict(model.named_parameters())

    x_batch = torch.randn(8, 10)
    y_batch = torch.randint(0, 5, (8,))

    penalty = get_sgo_penalty(params, model, x_batch, y_batch, class_aware=False)
    print(f"SGO Penalty: {penalty.item()}")
    assert penalty >= 0

    penalty_ca = get_sgo_penalty(params, model, x_batch, y_batch, class_aware=True)
    print(f"CSGO Penalty: {penalty_ca.item()}")
    assert penalty_ca >= 0

    # Backprop test
    penalty.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None
        print(f"Grad {name} norm: {p.grad.norm().item()}")

if __name__ == "__main__":
    test_sgo()
