import torch
import torch.nn as nn
from gsnr_adaptive_momentum_experiment.optimizer import GAM

def test_gam_step():
    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    gsnr = {p: torch.tensor([1.0, 0.5])}
    optimizer = GAM([p], lr=0.1, betas=(0.9, 0.999), weight_decay=0)

    p.grad = torch.tensor([0.1, 0.2])
    optimizer.step(gsnrs=gsnr)

    # Check that state is updated
    state = optimizer.state[p]
    assert state['step'] == 1
    assert 'exp_avg' in state
    assert 'exp_avg_sq' in state
    assert 'm_bias_corr' in state

    # First step with GSNR=1 should be like standard Adam
    # beta1_eff = 1 - (1-0.9)*1 = 0.9
    # m = 0.1 * 0.1 = 0.01
    # W = 0.1
    # m_hat = 0.1
    # v = 0.001 * 0.01 = 0.00001
    # v_hat = 0.01
    # step = 0.1 * 0.1 / 0.1 = 0.1
    # p = 1.0 - 0.1 = 0.9 (approx)

    assert p[0] < 1.0
    print("Test passed!")

if __name__ == "__main__":
    test_gam_step()
