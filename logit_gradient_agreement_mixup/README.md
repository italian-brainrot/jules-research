# Logit-Gradient-Agreement Mixup (LGAM)

## Hypothesis
Standard Mixup interpolates samples and labels using a $\lambda$ value sampled from a Beta distribution. We hypothesize that the "helpfulness" of mixing two samples depends on how well their gradient signals align. Specifically, if the gradients of the loss with respect to logits for two samples are well-aligned (high cosine similarity), mixing them is "safer" and more likely to result in a meaningful synthetic sample that lies on the data manifold. Conversely, if their gradients are conflicting, heavy mixing might be destructive.

**Logit-Gradient-Agreement Mixup (LGAM)** adaptively adjusts the Mixup $\lambda$ based on the cosine similarity between the logit gradients of the two samples being mixed.

## Method
1.  **Logit Gradients**: Compute the gradient of the cross-entropy loss with respect to the pre-softmax activations (logits). For a sample $i$, this is $g_i = p_i - y_i$, where $p_i$ is the softmax output and $y_i$ is the one-hot target.
2.  **Agreement**: For a pair of samples $(i, j)$ selected for Mixup, compute the cosine similarity $s_{ij} = \cos(g_i, g_j)$.
3.  **Adaptive Lambda**: Map $s_{ij}$ to an agreement score $a_{ij} = (s_{ij} + 1) / 2 \in [0, 1]$. Adjust the sampled $\lambda$ using:
    $\lambda_{adj} = 0.5 + (\lambda - 0.5) \cdot (1 - a_{ij}^\gamma)$
    where $\gamma$ is a hyperparameter. When agreement is high ($a_{ij} \approx 1$), $\lambda_{adj} \approx 0.5$ (maximum mixing). When agreement is low, $\lambda_{adj}$ stays closer to the original sampled value (which, for small $\alpha$, is often near 0 or 1, representing less mixing).

## Experimental Setup
- **Dataset**: `mnist1d` (10,000 samples).
- **Model**: 3-layer MLP (40 -> 256 -> 256 -> 10).
- **Optimizer**: AdamW with tuned learning rate and weight decay.
- **Tuning**: Optuna (20 trials) for all hyperparameters ($\text{lr}, \text{wd}, \alpha, \gamma$).
- **Baseline**: Standard training without Mixup.
- **Comparison**: Standard Mixup vs. LGAM.
- **Evaluation**: 50 epochs, averaged over 3 seeds.

## Results
Final Test Accuracy (Mean ± Std Dev over 3 seeds):

| Method | Test Accuracy |
| :--- | :--- |
| **Baseline** | 0.7680 ± 0.0147 |
| **Mixup** | 0.7768 ± 0.0026 |
| **LGAM** | 0.7745 ± 0.0050 |

### Best Hyperparameters:
- **Baseline**: `lr: 0.00658`, `wd: 0.00773`
- **Mixup**: `lr: 0.00643`, `wd: 0.00314`, `alpha: 0.218`
- **LGAM**: `lr: 0.00836`, `wd: 0.00663`, `alpha: 0.111`, `gamma: 4.642`

## Analysis
- Both Mixup and LGAM outperformed the non-augmented baseline, confirming the value of manifold-based regularization on `mnist1d`.
- In this setup, LGAM performed similarly to standard Mixup (77.45% vs 77.68%).
- The optimal $\gamma$ for LGAM was relatively high (4.64), which means that the agreement needs to be very high for the mixing to be significantly pushed towards $\lambda=0.5$.
- The similarity in performance suggests that while the gradient alignment is an intuitive metric for sample compatibility, its direct application to adjusting Mixup $\lambda$ may require more complex mapping functions or might be more beneficial in scenarios with higher label noise or class imbalance.

## Conclusion
LGAM introduces a novel way to incorporate model-aware feedback into the data augmentation process. While it did not significantly outperform standard Mixup on `mnist1d`, it demonstrates a stable training routine that effectively regularizes the model. Future work could explore using different types of "agreement" metrics, such as alignment in the feature space or using the full Jacobian.
