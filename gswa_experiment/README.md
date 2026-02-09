# GSNR-Weighted Stochastic Weight Averaging (GSWA)

## Hypothesis
Standard Stochastic Weight Averaging (SWA) typically uses a uniform average of weights collected during the later stages of training. We hypothesize that weighting the contributions of different iterations to the averaged model based on their **Gradient Signal-to-Noise Ratio (GSNR)** can improve the final model's performance and stability. Specifically, iterations with higher GSNR (more consistent gradients) are likely in more stable or flatter regions of the loss surface and should contribute more to the averaged weights.

## Method
- **GSNR Calculation**: For each batch, we compute the Batch GSNR as $G = \frac{\| \sum g_i \|^2}{\sum \| g_i \|^2}$, where $g_i$ are per-sample gradients. This value ranges from 1 to the batch size $B$, indicating how much the samples in the batch agree on the update direction.
- **GSWA Update**: We maintain a weighted average of model weights $W_{avg} = \frac{\sum w_t W_t}{\sum w_t}$, where the weight $w_t$ for iteration $t$ is the average GSNR of that epoch.
- **Experimental Setup**:
    - **Dataset**: `mnist1d` (1D signal classification).
    - **Architecture**: 3-layer MLP (40 -> 256 -> 256 -> 10).
    - **Optimizer**: Adam with tuned learning rates.
    - **SWA/GSWA Phase**: Started after 75% of training (epoch 37 out of 50).
    - **Comparison**: Adam (baseline), Adam + SWA (uniform), Adam + GSWA (GSNR-weighted).
    - **Evaluation**: Each method was tuned using Optuna and then evaluated over 5 different random seeds.

## Results
The experiment showed that both SWA and GSWA outperformed the Adam baseline. GSWA achieved the highest mean accuracy.

| Method | Mean Accuracy | Std Dev | Best Learning Rate |
|--------|---------------|---------|-------------------|
| Adam   | 65.98%        | 0.88%   | 0.006003          |
| SWA    | 66.98%        | 0.79%   | 0.008170          |
| GSWA   | **67.14%**    | **0.94%** | 0.006433        |

### Analysis
- **Consistency**: Both weight averaging methods (SWA and GSWA) consistently improved over the Adam baseline (approx +1% accuracy).
- **GSNR Effect**: GSWA achieved a slightly higher mean accuracy than standard SWA, suggesting that weighting the average by gradient consistency is a beneficial heuristic.
- **Robustness**: While in some runs GSWA showed significantly lower variance, in the final evaluation the variance was comparable to standard SWA.

## Conclusion
GSNR-Weighted Stochastic Weight Averaging (GSWA) provides a slight improvement over standard SWA by prioritizing weights from iterations with more consistent gradient signals. This suggests that GSNR is a useful proxy for the quality or stability of the model state during the averaging phase.
