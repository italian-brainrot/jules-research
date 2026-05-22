# Differentiable Frenet-Serret Signal Geometry Experiment

This experiment investigates whether local geometric invariants (curvature and torsion) of a signal's phase-space trajectory can improve classification performance on the `mnist1d` dataset.

## Hypothesis

A 1D signal $x(t)$ can be embedded into a higher-dimensional space to form a curve. We use a **delay embedding** $P(t) = (x(t), x(t-\tau), x(t-2\tau))$ where $\tau$ is a learnable delay. The local geometry of this curve, specifically its **curvature** ($\kappa$) and **torsion** ($\tau_{geo}$), captures the rate at which the signal changes direction and "twists" in phase space.

We hypothesize that these geometric features provide a useful inductive bias for capturing non-stationary oscillations and sharp transitions that might be difficult for a standard MLP to extract directly from raw signal values.

## Methodology

### 1. Differentiable Frenet-Serret Layer
- **Smoothing**: The input signal is first smoothed with a Gaussian kernel with a learnable $\sigma$. This ensures the signal is $C^3$ continuous, which is necessary for computing the third derivative.
- **Embedding**: The smoothed signal $s(t)$ is embedded into $\mathbb{R}^3$ via $P(t) = (s(t), s(t-\tau), s(t-2\tau))$. Fractional delays $\tau$ are handled using differentiable linear interpolation.
- **Geometric Invariants**:
    - **Curvature**: $\kappa = \frac{\|P' \times P''\|}{\|P'\|^3}$
    - **Torsion**: $\tau_{geo} = \frac{(P' \times P'') \cdot P'''}{\|P' \times P''\|^2}$
- **Feature Extraction**: We compute the mean, maximum, and standard deviation of both $\kappa(t)$ and $\tau_{geo}(t)$ across the signal, resulting in 6 global features.

### 2. Experimental Setup
- **Dataset**: `mnist1d` (10,000 samples).
- **Models**:
    - **Frenet-Serret Augmented MLP**: A 3-layer MLP that takes the raw signal concatenated with the 6 geometric features.
    - **Baseline MLP**: A 3-layer MLP with a slightly larger hidden dimension to match the parameter count of the augmented model.
- **Tuning**: Learning rates for both models were tuned using Optuna for 10 trials each.
- **Evaluation**: Final performance was evaluated over 5 random seeds for 50 epochs each.

## Results

| Model | Test Accuracy (Mean +/- Std) |
|---|---|
| Baseline MLP | 77.79% +/- 0.76% |
| **Frenet-Serret Augmented MLP** | **70.00% +/- 1.00%** |

### Analysis
- **Performance**: Contrary to our hypothesis, the Frenet-Serret augmented model performed significantly worse than the baseline MLP.
- **Redundancy and Noise**: It is possible that the curvature and torsion features are either redundant with what the MLP already learns or are too sensitive to noise, despite the Gaussian smoothing. The delay embedding parameters $\tau$ and $\sigma$ might also be difficult to optimize in the context of this specific classification task.
- **Global vs Local**: While we extracted global statistics (mean, max, std), the digits in `mnist1d` are characterized by localized spatial features. Averaging these geometric invariants across the entire signal may have discarded critical discriminative information.

## Conclusion

The Differentiable Frenet-Serret layer provides a mathematically sound way to incorporate differential geometry into neural networks. However, for the `mnist1d` digit classification task, these features did not provide a performance benefit. Future work could explore using the full sequences of $\kappa(t)$ and $\tau_{geo}(t)$ with a convolutional or recurrent architecture, or applying this technique to tasks where global geometric structure is more important (e.g., physiological signal analysis).

## Visualizations
The training progress and comparison can be seen in `comparison.png`.
