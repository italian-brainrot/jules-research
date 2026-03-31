# Differentiable Singular Spectrum Analysis (SSA) Experiment

This experiment evaluates a Differentiable SSA layer for signal denoising and feature extraction on a noisy version of the MNIST-1D dataset.

## Method

SSA is a classical time-series analysis technique that decomposes a signal into several interpretable components (trend, oscillatory, noise) using Singular Value Decomposition (SVD) on a Hankel matrix.

In this differentiable version:
1. The input signal $x$ is converted into a Hankel matrix $H$.
2. SVD is performed on $H = U S V^T$.
3. Learnable weights are applied to the singular values: $S_{weighted} = S \cdot \sigma(w)$.
4. The matrix is reconstructed and diagonal-averaged to obtain the filtered signal.
5. The entire process is differentiable, allowing the model to learn which singular components to preserve for the downstream task.

## Dataset
- **MNIST-1D** with added Gaussian noise ($\sigma=0.2$).
- Training samples: 10,000.
- Sequence length: 40.

## Results

Hyperparameters were tuned using Optuna (15 trials each). The models were then evaluated over 5 different seeds.

| Model | Test Accuracy (Noisy) |
|-------|-----------------------|
| Baseline MLP | 65.24% ± 0.97% |
| SSA + MLP | 64.14% ± 0.32% |

### Hyperparameters
- **Baseline MLP**: Learning Rate = 0.04314
- **SSA + MLP**: Learning Rate = 0.02128, Window Size = 10

## Discussion

The Differentiable SSA layer achieved a slightly lower mean accuracy than the baseline MLP but showed significantly lower variance across seeds (0.32% vs 0.97%). This suggests that the SSA layer provides a more stable feature representation by acting as a learnable filter, even if it doesn't improve the peak performance in this specific configuration. The window size and rank-weighting mechanism could be further explored for more complex or noisier signals where classical denoising is traditionally more effective.
