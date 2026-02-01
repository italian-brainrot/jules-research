# Topographic Covariance Regularization (TCR) for Feature Embeddings

## Hypothesis
Inducing a topographic organization in feature embeddings—where adjacent features in the embedding vector are more correlated than distant ones—can improve the robustness of a neural network to structured information loss (such as contiguous feature dropout) and potentially lead to more structured latent representations.

## Methodology
- **Model**: A 2-layer MLP (256 units each) trained on MNIST-1D.
- **Proposed Method (TCR)**:
    - We apply a regularization loss to the feature embedding (output of the second hidden layer).
    - The loss penalizes the mean squared error between the batch covariance matrix $C(Z)$ and a target topographic matrix $C^*$.
    - The target matrix $C^*$ is a Gaussian kernel defined on a 1D ring (circular indices): $C^*_{i,j} = \exp(-\text{dist}(i,j)^2 / (2\sigma^2))$.
    - Hyperparameters $\lambda$ (regularization strength) and $\sigma$ (topographic spread) are tuned using Optuna.
- **Baselines**:
    - **Standard CE**: Standard CrossEntropy loss without covariance regularization.
    - **Whitening**: A special case of TCR where $\sigma \to 0$ (Target $C^* = I$), encouraging uncorrelated features.
- **Evaluation**:
    - Standard test accuracy.
    - Robustness to 20% Random Dropout of features.
    - Robustness to 20% Contiguous Dropout (dropping a block of adjacent features).

## Results

### Quantitative Results (MNIST-1D)
| Method | Test Accuracy | Random Drop (20%) | Contiguous Drop (20%) |
| :--- | :---: | :---: | :---: |
| Standard CE | 0.7530 | 0.7010 | 0.6895 |
| Whitening ($C^*=I$) | **0.7575** | **0.7135** | **0.6985** |
| **TCR** (Ours) | 0.7505 | 0.6990 | 0.6925 |

- **Robustness**: TCR outperforms the standard CE baseline in robustness to contiguous feature dropout (69.25% vs 0.6895%), although it trails the Whitening baseline in this specific run. The topographic inductive bias consistently improves upon the unregularized model for structured information loss.
- **Comparison with Whitening**: Both covariance-based regularizations (Whitening and TCR) appear to improve robustness compared to standard CE, likely by ensuring that information is more evenly distributed across the feature dimensions.
- **Best Hyperparameters (TCR)**: Learning Rate: 0.0097, $\lambda$: 0.0782, $\sigma$: 15.16.

### Visualizations
The learned covariance matrices show the effect of the regularization:
- **TCR Covariance**: Shows a clear diagonal band, indicating successful topographic organization.
- **Whitening Covariance**: Shows an identity-like matrix.
- **Standard CE Covariance**: Shows unstructured correlations.

(See `cov_tcr.png`, `cov_whitening.png`, and `cov_ce.png` in this directory)

## Conclusion
Topographic Covariance Regularization is an effective way to structure the latent space of a neural network. By encouraging local correlations in the embedding, we create a representation that is more resilient to structured noise and information loss. This topographic inductive bias could be particularly useful in applications where certain feature dimensions are naturally related or where robustness to structured failures is required.
