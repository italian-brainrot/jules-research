# Momentum-Aligned Sample Weighting (MASW) Experiment

## Hypothesis
This experiment investigates **Momentum-Aligned Sample Weighting (MASW)**, a technique that dynamically weights per-sample gradients based on their alignment with the optimizer's momentum buffer.

The core hypothesis is that gradients that align well with the historical update direction (as captured by momentum) represent more reliable "consensus" information, while gradients that strongly conflict with the momentum may be noisy or represent outliers. By up-weighting aligned samples and down-weighting misaligned ones, we aim to improve optimization stability and generalization.

Specifically, for a sample $i$, we compute the cosine similarity $s_i$ between its gradient $g_i$ and the current momentum $m$. We then assign a weight $w_i \propto \exp(\gamma \cdot s_i)$ and update using the weighted average gradient.

## Methodology
- **Dataset**: `mnist1d` (4,000 samples).
- **Model**: A 3-layer MLP (40 -> 256 -> 256 -> 10) with ReLU activations.
- **Optimizers Compared**:
  - **Baseline**: Standard AdamW.
  - **MASW-AdamW**: AdamW where per-sample gradients are weighted by momentum alignment before the update step.
- **Hyperparameter Tuning**: Optuna was used to tune the learning rate, weight decay, and the alignment sensitivity $\gamma$ (10 trials per mode).
- **Evaluation**: The best-found hyperparameters were used for a final evaluation over 30 epochs.

## Results

| Method | Mean Test Accuracy | Best Hyperparameters |
|--------|-------------------|----------------------|
| **Baseline (AdamW)** | 62.63% | `lr`: 5.50e-3, `wd`: 2.62e-6 |
| **MASW-AdamW** | 62.63% | `lr`: 4.39e-3, `wd`: 1.99e-4, $\gamma$: 0.65 |

### Analysis
- In this experiment, MASW-AdamW achieved a mean test accuracy identical to the baseline AdamW (62.63%).
- During tuning, MASW showed some sensitivity to the $\gamma$ parameter. High $\gamma$ values (e.g., > 5) led to significantly worse performance (accuracy dropping to ~15-20%), suggesting that over-weighting based on momentum alignment can lead to "momentum collapse" or premature convergence to a narrow subspace.
- The optimal $\gamma \approx 0.65$ suggests that a mild preference for aligned gradients is preferred, but in this specific task, it didn't provide a significant boost over the standard uniform weighting.

## Conclusion
The hypothesis that momentum alignment can be used to distinguish between reliable and noisy samples was not strongly supported for the `mnist1d` task with an MLP. While the method is stable for small $\gamma$, it did not outperform a well-tuned AdamW baseline. This might be because `mnist1d` gradients are already relatively stable, or because the momentum itself is an imperfect proxy for "reliability" in this context. Future work could explore MASW on datasets with higher label noise, where the distinction between "signal" and "noise" gradients might be more pronounced.

## Visualizations
The `comparison.png` plot shows the training loss and test accuracy over 30 epochs for both methods using their best hyperparameters.
