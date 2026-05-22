import torch
import torch.nn as nn
import torch.nn.functional as F

class DPRSALayer(nn.Module):
    """
    Differentiable Phase-Rectified Signal Averaging (DPRSA) Layer.

    Learns to identify 'anchor points' in a 1D signal and computes a
    weighted average of segments centered at these points.
    """
    def __init__(self, in_channels, num_anchors=4, window_size=20, anchor_kernel_size=5, softmax_anchors=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.window_size = window_size
        self.softmax_anchors = softmax_anchors

        # Filter to compute anchorness scores
        self.anchor_filter = nn.Conv1d(
            in_channels,
            num_anchors,
            kernel_size=anchor_kernel_size,
            padding=anchor_kernel_size // 2
        )

    def forward(self, x):
        # x: (batch, in_channels, length)
        batch_size, channels, length = x.shape

        # Compute anchor scores
        # scores: (batch, num_anchors, length)
        scores = self.anchor_filter(x)

        if self.softmax_anchors:
            # Softmax over time dimension to pick the most prominent anchor(s)
            scores = F.softmax(scores, dim=-1)
        else:
            # Sigmoid to allow multiple anchor points
            scores = torch.sigmoid(scores)

        # Pad x to allow window extraction at edges
        # We want window_size around each t.
        # To get exactly 'length' windows of size 'window_size':
        # (L + pad_left + pad_right - window_size) + 1 = L
        # pad_left + pad_right = window_size - 1
        pad_left = self.window_size // 2
        pad_right = self.window_size - 1 - pad_left
        x_padded = F.pad(x, (pad_left, pad_right), mode='replicate')

        # Use unfold to get all windows
        # x_unfolded: (batch, channels, length, window_size)
        x_unfolded = x_padded.unfold(2, self.window_size, 1)

        # We want to compute:
        # out[b, a, c, w] = sum_t scores[b, a, t] * x_unfolded[b, c, t, w] / sum_t scores[b, a, t]

        # scores: (batch, num_anchors, length)
        # x_unfolded: (batch, channels, length, window_size)

        # Reshape scores for einsum: (batch, num_anchors, length)
        # Reshape x_unfolded for einsum: (batch, channels, length, window_size)

        # Weighted sum over length (t)
        # out: (batch, num_anchors, channels, window_size)
        out = torch.einsum('bal, bclw -> bacw', scores, x_unfolded)

        # Normalization
        score_sums = scores.sum(dim=-1, keepdim=True).unsqueeze(2) # (batch, num_anchors, 1, 1)
        out = out / (score_sums + 1e-8)

        # Flatten num_anchors and channels for the next layer if needed,
        # or keep them separate. Let's return (batch, num_anchors * channels * window_size)
        return out.view(batch_size, -1)

if __name__ == "__main__":
    # Quick test
    layer = DPRSALayer(in_channels=1, num_anchors=2, window_size=10)
    x = torch.randn(8, 1, 40)
    out = layer(x)
    print(out.shape) # Should be (8, 2 * 1 * 10) = (8, 20)
