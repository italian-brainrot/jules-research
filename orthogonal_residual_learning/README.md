# Orthogonal Residual Learning (ORL)

## Hypothesis
In standard Residual Networks (ResNets), each block learns a residual function $f(x)$ such that the next state is $x + f(x)$. However, there is no explicit constraint preventing $f(x)$ from being redundant with $x$ (e.g., $f(x) \approx \alpha x$).

We hypothesize that forcing the residual branch $f(x)$ to be orthogonal to its input $x$ will encourage the model to learn more diverse and complementary features, leading to better generalization and potentially faster convergence or higher peak performance.

We explore two ways to enforce this:
1.  **Penalty-based ORL**: Adding a squared cosine similarity penalty between $x$ and $f(x)$ to the loss function.
2.  **Forced ORL (Forced-Orth)**: Explicitly projecting $f(x)$ onto the orthogonal complement of $x$ during the forward pass: $f_{orth}(x) = f(x) - \frac{x \cdot f(x)}{||x||^2} x$.

## Methodology
- **Dataset**: `mnist1d` (10,000 samples, 40 input features, 10 classes).
- **Model Architecture**:
    - Input Layer: 40 -> 128
    - 4 Residual Blocks (128 -> 128 each)
    - Output Layer: 128 -> 10
- **Variants**:
    - **Baseline**: Standard ResMLP ($x + f(x)$).
    - **Penalty**: $x + f(x)$ with loss $L = L_{CE} + \lambda \sum \text{cos}^2(x, f(x))$.
    - **Forced**: $x + f_{orth}(x)$.
- **Optimization**:
    - Optuna used for tuning learning rate (all variants) and $\lambda$ (Penalty variant).
    - 15 trials per variant.
    - Final training for 50 epochs using best hyperparameters.

## Results

| Variant  | Best Test Accuracy | Learning Rate | $\lambda_{orth}$ |
|----------|-------------------|---------------|------------------|
| Baseline | 80.95%            | 1.13e-3       | N/A              |
| Penalty  | 82.00%            | 1.61e-3       | 1.27e-4          |
| **Forced** | **82.15%**        | 1.29e-3       | N/A              |

### Analysis
- **Performance Improvement**: Both the Penalty and Forced variants outperformed the baseline. The Forced variant achieved the highest accuracy (82.15%), an improvement of 1.2% over the baseline.
- **Feature Diversity**: By forcing orthogonality, we ensure that each layer adds information in a subspace that is not currently occupied by the input vector. This effectively forces the network to explore new feature dimensions at each step.
- **Network Adaptation**: In the Forced variant, we tracked the "natural" cosine similarity (the similarity before projection). Interestingly, the network adapted by producing residuals that were slightly negatively correlated with the input (around -0.04), whereas the baseline produced positively correlated residuals (around 0.10).
- **Regularization Effect**: The Penalty variant also showed strong performance, suggesting that even a soft constraint on orthogonality helps generalization.

## Visualizations
The file `results.png` contains plots for:
1. **Test Accuracy**: Both ORL variants consistently stay above the baseline after the initial phase.
2. **Cosine Similarity**: Shows how the baseline and penalty variants maintain certain levels of correlation, while the "forced" variant's unconstrained output actually becomes slightly anti-correlated.
3. **Cosine Similarity Squared**: Measures the "redundancy" between input and residual.
4. **Test Loss**: ORL variants generally achieve lower test loss.

## Conclusion
Orthogonal Residual Learning is a simple yet effective modification to the residual connection. By either penalizing or explicitly removing the component of the residual that is parallel to the input, we force the network to learn non-redundant, complementary features. This leads to improved classification performance on the `mnist1d` task.
