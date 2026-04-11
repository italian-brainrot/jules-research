# Differentiable Bispectrum Experiment

This experiment investigates the use of a **Differentiable Bispectrum Layer** as an inductive bias for 1D signal classification.

## Hypothesis

The bispectrum is a third-order statistic that captures phase relationships between different frequency components and is theoretically shift-invariant. Unlike the power spectrum (second-order), which discards all phase information, the bispectrum retains some phase information while being invariant to translation.

We hypothesize that augmenting a neural network with bispectrum features will improve its performance and robustness on 1D signal classification tasks (like `mnist1d`) by providing features that are naturally invariant to shifts and sensitive to non-linear couplings between frequencies.

## Methodology

### 1. Differentiable Bispectrum Layer
- **Definition**: The bispectrum $B(k, l)$ of a signal $X(f)$ is defined as $B(k, l) = X(k)X(l)X^*(k+l)$.
- **Implementation**: We implement a differentiable version using PyTorch's `torch.fft.rfft`. We compute the magnitude and phase of the bispectrum in the non-redundant region for a real-valued signal.
- **Invariance**: Verified the shift-invariance of the bispectrum magnitude through unit tests.

### 2. Experimental Setup
- **Dataset**: `mnist1d` (10,000 samples).
- **Models**:
    - **Baseline MLP**: 3-layer MLP (40 -> 256 -> 256 -> 10).
    - **Bispectrum-Augmented MLP**: The input signal is augmented with its bispectrum features (magnitude and phase), then passed through an MLP.
    - **Conv1d**: A standard 1D convolutional network.
- **Tuning**: Learning rates for all models were tuned using Optuna (10 trials, 20 epochs each).
- **Evaluation**: Final evaluation over 50 epochs with 3 different seeds.

## Results

| Model | Test Accuracy (Mean +/- Std) | Best LR |
|---|---|---|
| Baseline MLP | 81.27% +/- 1.06% | 0.003175 |
| **Bispectrum-Augmented MLP** | 67.17% +/- 0.12% | 0.000820 |
| Conv1d | 96.38% +/- 0.24% | 0.009992 |

Note: The Bispectrum-Augmented MLP performed significantly worse than the baseline MLP in this configuration.

## Analysis

- **Performance Gap**: Surprisingly, the bispectrum-augmented MLP performed worse than the baseline. This could be due to several factors:
    - **Feature Dimensionality**: The bispectrum adds a large number of features (121 for $N=40$), which might lead to overfitting or optimization difficulties when combined with raw inputs.
    - **Phase Information**: The phase of the bispectrum might be noisy or difficult for the MLP to utilize without proper scaling or normalization.
    - **Task Suitability**: While `mnist1d` involves shifts, it also contains other variations (scaling, noise) where the exact bispectrum might not be as invariant or discriminative as learned convolutional filters.
- **Stability**: The bispectrum model showed very low standard deviation across seeds (0.12%), suggesting that it might provide a very stable (albeit less accurate) representation.

## Conclusion

The Differentiable Bispectrum Layer, as implemented and used in this experiment, did not provide the expected performance boost for `mnist1d` classification. While theoretically appealing due to its shift-invariance and ability to capture higher-order statistics, its practical integration into deep learning models for this specific task requires more careful architectural consideration (e.g., feature selection, normalization, or specific bispectrum-based pooling).

## Visualizations

- `comparison.png`: Comparison of model accuracies.
- `results.txt`: Raw numerical results.
