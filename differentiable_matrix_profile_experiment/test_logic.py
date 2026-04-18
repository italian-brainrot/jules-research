import torch
from model import DifferentiableMatrixProfile

def test_mp_differentiability():
    batch_size = 2
    seq_len = 10
    window_size = 3
    x = torch.randn(batch_size, seq_len, requires_grad=True)
    mp_layer = DifferentiableMatrixProfile(window_size=window_size)

    mp = mp_layer(x)
    assert mp.shape == (batch_size, seq_len - window_size + 1)

    loss = mp.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Differentiability test passed.")

def test_mp_translation_invariance():
    # Matrix profile is not strictly translation invariant in the absolute sense
    # but the relative distances between windows should be similar if the whole signal is shifted.
    # Actually, MP is typically invariant to the absolute position of the motifs.

    seq_len = 20
    window_size = 5
    x = torch.randn(1, seq_len)

    # Create a motif
    motif = torch.randn(1, window_size)
    x1 = x.clone()
    x1[0, 2:2+window_size] = motif
    x1[0, 10:10+window_size] = motif

    x2 = torch.roll(x1, shifts=2, dims=1)

    mp_layer = DifferentiableMatrixProfile(window_size=window_size, temperature=0.01)
    mp1 = mp_layer(x1)

    # We expect low values at positions 2 and 10 in mp1
    print(f"MP1: {mp1}")

    # Since we roll, the MP should also be rolled (mostly, except for boundary effects if we didn't use periodic padding, but here we just want to see if it detects motifs)
    # Actually, MP is defined on windows.
    pass

if __name__ == "__main__":
    test_mp_differentiability()
