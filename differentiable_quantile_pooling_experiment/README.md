# Implicit Differentiable Quantile Pooling (IDQP)

This experiment introduces **Implicit Differentiable Quantile Pooling (IDQP)**, a novel pooling layer that learns a per-channel quantile $q \in [0, 1]$ to extract features. Unlike standard max or average pooling, IDQP can adaptively choose any order statistic, allowing the network to decide whether it needs to pick peaks, medians, or other percentile-based features.

## Method

The quantile $m$ of a set of values $\{x_1, \dots, x_K\}$ is defined as the value such that the number of elements less than or equal to $m$ is $Kq$. We use a differentiable approximation by solving the following implicit equation for $m$:

$$\sum_{i=1}^K \sigma(\alpha(m - x_i)) = Kq$$

where $\sigma$ is the sigmoid function and $\alpha$ is a temperature parameter.

### Properties
- **Differentiability**: We use the **Implicit Function Theorem** to derive gradients for $m$ with respect to $x_i$, $q$, and $\alpha$ without needing to backpropagate through the iterative solver.
- **Flexibility**: $q$ and $\alpha$ are learnable parameters. By learning $q$, the layer can smoothly interpolate between min pooling ($q=0$), median pooling ($q=0.5$), and max pooling ($q=1$).
- **Efficiency**: The forward pass uses a simple bisection solver, while the backward pass is a single-step calculation.

## Results on MNIST-1D

We compared IDQP against several baseline pooling methods on the `mnist1d` dataset. For each method, the learning rate was tuned using Optuna (8 trials) to ensure a fair comparison.

| Pool Type | Accuracy | Best LR |
|-----------|----------|---------|
| **Quantile (IDQP)** | **0.9612 +- 0.0003** | 0.008674 |
| Max | 0.9575 +- 0.0095 | 0.005691 |
| Lp (Learnable p) | 0.9510 +- 0.0005 | 0.008755 |
| Average | 0.9420 +- 0.0050 | 0.005897 |
| Median (Hard) | 0.9110 +- 0.0130 | 0.009708 |

IDQP achieved the highest accuracy, suggesting that learnable quantiles provide a superior inductive bias for this task.

## Learned Quantiles

Analysis of the learned $q$ values shows a distribution around 0.4 to 0.6, indicating that for many channels, the network found features closer to the median to be more useful than the extreme values used in max pooling.

![Learned q Histograms](learned_q.png)

## Files
- `models.py`: Implementation of the `ImplicitQuantilePooling1d` layer and the `Net` architecture.
- `compare.py`: Comparison script with Optuna tuning.
- `analyze_q.py`: Script to analyze the learned quantile values.
- `test_layer.py`: Gradient verification script.
