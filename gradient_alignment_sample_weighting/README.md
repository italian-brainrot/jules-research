# Gradient-Alignment Sample Weighting (GASW) Experiment

## Hypothesis
We hypothesize that **Gradient-Alignment Sample Weighting (GASW)** improves generalization by dynamically weighting each sample's contribution to the batch update based on its alignment with the common batch direction.

In stochastic gradient descent, every sample in a batch typically contributes equally to the update. However, some samples might be outliers, have noisy labels, or simply be "confusing" in the current state of the model, leading to gradients that point in directions inconsistent with the general consensus of the batch. By down-weighting these inconsistent samples and focusing on those that align with the batch-average gradient, we expect to obtain a cleaner, more robust training signal that leads to better generalization.

## Methodology
- **Algorithm**:
    1. For each batch, compute per-sample gradients $g_i$ using `torch.func.vmap`.
    2. Compute the batch-average gradient $\bar{g} = \frac{1}{B} \sum g_i$.
    3. Calculate the cosine similarity $s_i$ between each $g_i$ and $\bar{g}$.
    4. Compute weights $w_i = \max(0, s_i)^\gamma$.
    5. Normalize weights such that $\sum w_i = B$.
    6. Update the model using the weighted average gradient $g = \frac{1}{B} \sum w_i g_i$.
- **Baseline**: Standard Adam optimizer.
- **Comparison**:
    - **GASW**: Gradient-Alignment Sample Weighting (emphasizes consensus).
    - **GDSW**: Gradient-Diversity Sample Weighting, where $w_i = (1 - s_i)^\gamma$ (emphasizes diversity/disagreement).
- **Dataset**: `mnist1d` with 4000 samples.
- **Model**: 3-layer MLP (40 -> 256 -> 256 -> 10).
- **Hyperparameter Tuning**: Optuna was used to tune the learning rate for all modes and the exponent $\gamma$ for GASW and GDSW.

## Results

| Mode | Best Hyperparameters | Final Test Accuracy | Final Training Loss |
| :--- | :--- | :--- | :--- |
| **Baseline** | `lr`: 0.0039 | 59.62% | 0.0015 |
| **GASW (Alignment)** | `lr`: 0.0045, `gamma`: 2.09 | **61.12%** | 0.2449 |
| **GDSW (Diversity)** | `lr`: 0.0060, `gamma`: 2.11 | 59.38% | 0.0031 |

### Analysis
- **GASW outperformed the baseline by 1.5% absolute accuracy.** This supports the hypothesis that prioritizing consensus in gradients can lead to better generalization.
- **GASW acted as a strong regularizer.** While the baseline reached a very low training loss (0.0015), GASW maintained a much higher loss (0.2449) while achieving higher test accuracy. This suggests that GASW prevents the model from overfitting to individual sample "noise" or outliers that disagree with the batch consensus.
- **GDSW (Diversity) did not improve over the baseline.** This is interesting as it contrasts with some theories (like SGO) that suggest gradient diversity is always beneficial. In this specific task and model, focusing on the common signal was more effective than emphasizing unusual directions.
- **Robustness**: The large $\gamma \approx 2.1$ found for GASW indicates that being quite selective about which samples to trust is beneficial.

## Conclusion
Gradient-Alignment Sample Weighting (GASW) is an effective dynamic regularization technique that improves generalization by filtering the training signal based on intra-batch consensus. By automatically down-weighting samples that "disagree" with the batch average, it prevents overfitting to non-representative gradients and focuses the learning process on robust, shared features.

## Visualizations
The training loss and test accuracy curves are shown in `comparison.png`.
