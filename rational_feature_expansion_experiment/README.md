# Rational Feature Expansion (RFE) Experiment

This experiment investigates **Rational Feature Expansion (RFE)**, a non-linear model with a tractable exact (iterative but closed-form per step) solution.

## Methodology

### Model
The RFE model represents the target function as a ratio of two linear forms:
$$f(x) = \frac{P(x)}{Q(x)} = \frac{w_P^T \phi(x)}{1 + w_Q^T \phi(x)}$$
where $\phi(x)$ are **Random Fourier Features (RFF)**.

### Training (IRLS)
To solve for $w_P$ and $w_Q$ exactly in each step, we use the **Sanathanan-Koerner (SK) iteration**, also known as Iterative Rational Least Squares. In each step $k$, we solve the weighted linear least squares problem:
$$\min_{w_P, w_Q} \sum_i \left| \frac{y_i Q_k(x_i) - P_k(x_i)}{Q_{k-1}(x_i)} \right|^2$$
where $Q_{k-1}$ is the denominator from the previous iteration. The first iteration starts with $Q_0 = 1$ (which is the standard Levy's method).

### Baselines
1.  **Extreme Learning Machine (ELM)**: A linear model over the same RFF features (equivalent to RFE with $Q(x)=1$).
2.  **Multi-Layer Perceptron (MLP)**: A standard neural network with tuned hyperparameters.

## Results on MNIST-1D

We compared the models on the `mnist1d` dataset (4000 samples) with hyperparameter tuning via Optuna.

| Model | Best Test Accuracy |
| :--- | :--- |
| **ELM** | 53.37% |
| **RFE** | 14.75% |
| **MLP** | 61.25% |

### Observations
- **RFE Failure**: The RFE model performed significantly worse than the linear ELM baseline. This is likely due to the inherent instability of rational functions (poles) when applied to one-hot classification targets.
- **Overfitting/Instability**: The iterative process in IRLS can lead to denominators $Q(x)$ that are very close to zero for some points, causing extreme values in predictions and poor generalization.
- **Inductive Bias**: Rational functions might be more suitable for regression tasks with smooth but sharply changing behaviors than for high-dimensional discrete classification.

## How to run
To run the comparison script:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 rational_feature_expansion_experiment/compare.py
```
