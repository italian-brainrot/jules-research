# Differentiable LPC Experiment Results

This experiment evaluates the effectiveness of using Linear Predictive Coding (LPC) coefficients as features for signal classification on the MNIST-1D dataset. We compare two differentiable implementations of LPC: Burg's method and Levinson-Durbin recursion.

## Method
LPC is a powerful tool in signal processing, especially for speech analysis, where it captures the spectral envelope of a signal by modeling it as the output of a linear all-pole filter. Here, we implemented two common ways to estimate these coefficients in a fully differentiable manner using PyTorch:

1.  **Levinson-Durbin Recursion**: Computes LPC coefficients from the signal's autocorrelation function.
2.  **Burg's Method**: Directly estimates reflection coefficients by minimizing the sum of forward and backward prediction errors, which typically provides better spectral resolution and is more stable than Levinson-Durbin.

The model architecture consists of an LPC layer that extracts `order` coefficients from the 40-dimensional MNIST-1D signals, followed by a standard MLP classifier.

## Comparison Results

| Model | Best Test Accuracy (20 epochs) |
| --- | --- |
| Baseline MLP | 0.7730 |
| LPC-Burg + MLP | 0.3795 |
| LPC-Levinson + MLP | 0.3905 |

## Findings
- **Performance Gap**: The LPC-based models significantly underperform the Baseline MLP on MNIST-1D. This is likely because the LPC coefficients only capture the spectral envelope (magnitudes of the poles) and discard all phase information, which is critical for distinguishing the shifted and distorted shapes in MNIST-1D.
- **Order Sensitivity**: Tuning showed that higher orders (up to 30) were often tried, but lower orders sometimes performed more stably during early training.
- **Differentiability**: Both implementations are fully differentiable, and the `test_logic.py` script confirmed that gradients flow back to the input signal without numerical issues, even with the recursive nature of the algorithms.
- **Stability**: Burg's method, while theoretically more stable, did not show a clear performance advantage over Levinson-Durbin in this specific classification task.

While LPC coefficients are highly effective for speech coding and synthesis, they may be too lossy as a standalone feature extraction layer for general 1D signal classification tasks like MNIST-1D without being combined with other features that preserve temporal or phase information.
