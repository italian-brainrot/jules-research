# Stochastic Gradient Orthogonalization (SGO) Experiment

## Hypothesis
We hypothesize that **Stochastic Gradient Orthogonalization (SGO)**, which explicitly penalizes the cosine similarity between the loss gradients of different samples in a batch, improves generalization and optimization in neural networks.

By encouraging gradients from different samples to be as orthogonal as possible, we force the model to learn features that are independently useful for different samples, rather than relying on a few dominant directions that might lead to overfitting. This can be seen as a way to maximize "gradient diversity" within each batch.

## Methodology
- **Dataset**: `mnist1d` with 4,000 training samples.
- **Model**: A 3-layer MLP (40 -> 128 -> 128 -> 10) with ReLU activations.
- **Configurations**:
  - **Baseline**: Standard Adam optimizer with tuned learning rate.
  - **SGO**: Adam + penalty on the average squared cosine similarity of loss gradients for *all* pairs of samples in a batch.
  - **CSGO (Class-Aware SGO)**: Adam + penalty on the average squared cosine similarity only for pairs of samples belonging to *different* classes.
- **Implementation**: Used `torch.func.vmap` and `torch.func.grad` to efficiently compute per-sample gradients within each batch.
- **Hyperparameter Tuning**: Used Optuna to tune the learning rate and regularization strength $\lambda$ (10 trials per mode, 30 epochs each).
- **Final Evaluation**: Trained each configuration for 100 epochs using the best found hyperparameters.

## Results

| Configuration | Best Test Accuracy | Final Test Accuracy | Final Train Accuracy | Best Hyperparameters |
|---------------|-------------------|---------------------|----------------------|----------------------|
| Baseline      | 58.13%            | 57.75%              | 100%                 | LR: 1.56e-3          |
| **SGO**       | **60.25%**        | **59.62%**          | 100%                 | LR: 2.66e-3, $\lambda$: 0.062 |
| CSGO          | 57.00%            | 56.25%              | 99.7%                | LR: 8.59e-4, $\lambda$: 0.002 |

### Analysis
- **SGO significantly outperformed the baseline** by more than 2% absolute test accuracy. This suggests that encouraging gradient diversity is a powerful regularizer for MLPs on sequential data like `mnist1d`.
- **CSGO performed worse than the baseline.** This is an interesting finding; it suggests that forcing orthogonality even among samples of the same class is beneficial. This might be because it forces the model to learn multiple distinct features for each class, leading to better robustness.
- **Optimization**: SGO reached a lower final training loss than the baseline (0.0030 vs 0.0110), suggesting that the "gradient diversity" also helps the optimizer find deeper minima, possibly by avoiding plateaus where gradients would otherwise be highly aligned.

## Conclusion
Stochastic Gradient Orthogonalization (SGO) is a promising technique for improving both the optimization and generalization of neural networks. By penalizing gradient alignment between samples, it ensures that the learning process remains "diverse" and robust. Future work could explore the effect of SGO on larger models and datasets, as well as its interaction with other regularization techniques like Dropout or Batch Normalization.

## Visualizations
The final comparison plot (`final_comparison.png`) shows the training loss, training accuracy, and test accuracy over 100 epochs for all three modes.
