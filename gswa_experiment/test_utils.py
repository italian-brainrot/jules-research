import torch
from gswa_experiment.model import MLP
from gswa_experiment.utils import compute_per_sample_grads_and_loss, compute_batch_gsnr

def test_gsnr_calculation():
    model = MLP(input_size=10, hidden_size=20, output_size=5)
    x = torch.randn(8, 10)
    y = torch.randint(0, 5, (8,))

    grads, loss = compute_per_sample_grads_and_loss(model, x, y)
    assert len(grads) > 0
    assert isinstance(loss, float)
    for name, g in grads.items():
        assert g.shape[0] == 8

    gsnr = compute_batch_gsnr(grads)
    print(f"Computed GSNR: {gsnr}")
    assert gsnr > 0

if __name__ == "__main__":
    test_gsnr_calculation()
    print("Tests passed!")
