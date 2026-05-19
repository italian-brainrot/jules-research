import torch
import torch.nn.functional as F
from layer import DPRSALayer

def test_translation_invariance():
    print("Testing translation invariance (approximate)...")
    batch_size = 2
    in_channels = 1
    length = 40
    window_size = 10
    num_anchors = 2

    layer = DPRSALayer(in_channels, num_anchors, window_size)
    x = torch.randn(batch_size, in_channels, length)

    # Standard output
    out1 = layer(x)

    # Roll the input
    shift = 5
    x_rolled = torch.roll(x, shifts=shift, dims=-1)
    out2 = layer(x_rolled)

    # Note: PRSA is expected to be more translation invariant than the raw signal
    # if the anchor points shift with the signal.
    # However, because we use a Conv1d to find anchors, and it's not perfectly invariant
    # (due to padding/edges), and the weighted average is over the whole length,
    # it should be relatively stable.

    # Let's check if the change in output is smaller for PRSA than for raw signal
    raw_diff = (x - x_rolled).pow(2).mean()
    prsa_diff = (out1 - out2).pow(2).mean()

    print(f"Raw mean squared difference: {raw_diff.item():.6f}")
    print(f"PRSA mean squared difference: {prsa_diff.item():.6f}")

    # This is not a strict test of invariance, but a check of relative stability.
    # In some cases PRSA might actually be perfectly invariant if the signal is periodic.

def test_gradients():
    print("Testing gradient flow...")
    batch_size = 4
    in_channels = 1
    length = 40
    window_size = 10
    num_anchors = 2

    layer = DPRSALayer(in_channels, num_anchors, window_size)
    x = torch.randn(batch_size, in_channels, length, requires_grad=True)

    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()

    if x.grad is not None:
        print("Gradient w.r.t input: OK")
    else:
        print("Gradient w.r.t input: FAILED")

    has_param_grad = True
    for name, param in layer.named_parameters():
        if param.grad is None:
            print(f"Gradient w.r.t {name}: FAILED")
            has_param_grad = False

    if has_param_grad:
        print("Gradients w.r.t parameters: OK")

def test_softmax_anchors():
    print("Testing softmax anchors...")
    layer = DPRSALayer(in_channels=1, num_anchors=2, window_size=10, softmax_anchors=True)
    x = torch.randn(8, 1, 40)
    out = layer(x)
    print(f"Output shape with softmax anchors: {out.shape}")
    if out.shape == (8, 20):
        print("Shape: OK")
    else:
        print(f"Shape: FAILED (got {out.shape})")

if __name__ == "__main__":
    test_translation_invariance()
    test_gradients()
    test_softmax_anchors()
