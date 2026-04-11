# Differentiable Hilbert Transform Experiment

This experiment investigates whether augmenting a standard Multi-Layer Perceptron (MLP) with features derived from the **Hilbert Transform** (instantaneous envelope and instantaneous frequency) improves performance on 1D signal classification (MNIST-1D).

## Hypothesis
Explicitly providing the model with instantaneous envelope and frequency features, extracted via a differentiable Hilbert transform, should provide a more expressive representation of the signal's local dynamics and improve classification accuracy.

## Methodology
- **Differentiable Hilbert Layer**: Implemented using `torch.fft` to compute the analytic signal.
  - **Instantaneous Envelope**: Computed as the magnitude of the analytic signal.
  - **Instantaneous Frequency**: Estimated as the phase difference between consecutive samples of the analytic signal.
- **Models**:
  - **Baseline MLP**: A 2-layer MLP with BatchNorm, processing the raw 40-dimensional signal.
  - **Hilbert MLP**: A 2-layer MLP with BatchNorm, processing the concatenation of the raw signal, its envelope, and its frequency (120-dimensional input).
- **Dataset**: MNIST-1D (10,000 samples).
- **Hyperparameter Tuning**: Learning rates for both models were tuned using Optuna (5 trials each).
- **Evaluation**: Best configurations were evaluated over 3 random seeds for 50 epochs each.

## Results

| Model | Best LR | Test Accuracy (Mean +/- Std) |
| :--- | :--- | :--- |
| **Baseline MLP** | 0.00414 | **80.53% +/- 0.45%** |
| **Hilbert MLP** | 0.00484 | 79.80% +/- 0.48% |

## Analysis
The `Hilbert MLP` did not outperform the `Baseline MLP`, despite having more parameters in its first layer (due to the tripled input dimension).

Possible reasons:
1. **Signal Length**: MNIST-1D signals are very short (40 samples). Hilbert-based features like instantaneous frequency and envelope might be more beneficial for longer, more complex non-stationary signals where local oscillations are more prominent.
2. **Redundancy**: The MLP might already be capable of learning similar features from the raw signal if they are useful for the task.
3. **Feature Scale/Distribution**: The instantaneous frequency and envelope might have different distributions that could make optimization slightly more challenging, although BatchNorm was used to mitigate this.

## Conclusion
While the Differentiable Hilbert Transform is a powerful tool for signal analysis, it did not provide an advantage for the MNIST-1D classification task in this specific architecture. This suggests that for short signals like these, standard MLPs are already quite efficient at extracting the necessary discriminative features.

## Verification
Mathematical correctness and differentiability of the `DifferentiableHilbertLayer` were verified in `test_logic.py`, including checks on the envelope and frequency of known sine waves.
