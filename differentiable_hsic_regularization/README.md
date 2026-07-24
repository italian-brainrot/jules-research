# Differentiable HSIC Bottleneck Regularization

This experiment explores the use of the Hilbert-Schmidt Independence Criterion (HSIC) as a bottleneck regularizer in deep neural networks.

## Hypothesis

By using HSIC to minimize the dependency between the input $X$ and a hidden representation $H$ (compression) while maximizing the dependency between $H$ and the targets $Y$ (sufficiency), we can encourage the model to learn more robust and generalizable features, similar to the Information Bottleneck principle but using a differentiable kernel-based dependency measure.

## Methodology

1.  **HSIC Implementation**: A differentiable version of HSIC was implemented using the RBF kernel. The median heuristic was used for the kernel bandwidth.
2.  **Models**:
    *   **BaselineMLP**: A standard 3-layer MLP.
    *   **HSICRegularizedMLP**: The same MLP architecture, but with an additional loss term:
        $Loss = CE(Out, Y) + \lambda_{in} HSIC(H, X) - \lambda_{out} HSIC(H, Y)$
3.  **Dataset**: `mnist1d` (10,000 samples).
4.  **Tuning**: Optuna was used to tune the learning rate, weight decay, and HSIC penalty weights ($\lambda_{in}$, $\lambda_{out}$) for both models over 10 trials.
5.  **Evaluation**: Both models were evaluated using their best hyperparameters over 5 different random seeds.

## Results

| Model | Accuracy |
| :--- | :--- |
| **BaselineMLP** | 78.03% ± 0.23% |
| **HSIC-Regularized MLP** | **78.60% ± 0.55%** |

The HSIC-regularized model showed a slight but consistent improvement over the baseline, suggesting that kernel-based dependency regularization can be beneficial for feature learning in small-to-medium scale signal classification tasks.

## Artifacts

- `results.png`: Comparison of accuracy between Baseline and HSIC models.
- `results.txt`: Detailed numerical results and best hyperparameters found by Optuna.
