import torch
from models import ResBlock, ResMLP

def test_res_block_orthogonality():
    dim = 64
    x = torch.randn(32, dim)

    # Test forced variant
    block = ResBlock(dim, variant='forced')
    block.eval()
    x_fixed = torch.randn(32, dim)
    v_out = block(x_fixed)
    # The actual residual added is (v_out - x_fixed)
    residual = v_out - x_fixed

    dot = (x_fixed * residual).sum(dim=-1)
    assert torch.all(torch.abs(dot) < 1e-5), f"Residual not orthogonal to input, dot product: {dot}"

def test_res_mlp_forward():
    model = ResMLP(40, 128, 10, 4, variant='baseline')
    x = torch.randn(32, 40)
    out = model(x)
    assert out.shape == (32, 10)

    model = ResMLP(40, 128, 10, 4, variant='forced')
    out = model(x)
    assert out.shape == (32, 10)

    model = ResMLP(40, 128, 10, 4, variant='penalty')
    out = model(x)
    assert out.shape == (32, 10)
    loss = model.get_orth_loss()
    assert loss >= 0

if __name__ == "__main__":
    test_res_block_orthogonality()
    test_res_mlp_forward()
    print("Tests passed!")
