# Prepended Greedy Layer-wise Learning Experiment

## Overview
This experiment investigates a greedy layer-wise learning method where new layers are *prepended* to an existing model. The process starts with a linear model fitted via least squares. Then, for each new layer:
1. The inputs $Z$ to the current model are optimized to minimize the loss.
2. A new layer is prepended, with its weights $W$ and bias $b$ fitted via least squares such that $\sigma(W X + b) \approx Z$, where $\sigma$ is an invertible activation function.
3. The process is repeated.

We also tested a "Linear Constrained" variant where the optimization of $Z$ includes a regularization term $\lambda \|Z - X\|^2$ to encourage the mapping $X \to Z$ to be more linear.

## Experimental Setup
- **Dataset**: MNIST1D (10,000 samples).
- **Activation**: LeakyReLU (slope 0.1).
- **Baselines**:
    - Standard MLP trained with Adam and Cross-Entropy loss.
- **PGL Variants**:
    - **PGL Basic**: Optimization of $Z$ without constraints.
    - **PGL Linear Constrained**: Regularization with $\lambda=0.1$ and $\lambda=1.0$.

## Results

### Accuracy Comparison
| Number of Layers | PGL Basic | PGL Lin (0.1) | PGL Lin (1.0) | MLP (Adam) |
|------------------|-----------|---------------|---------------|------------|
| 1 (Linear)       | 24.20%    | 24.20%        | 24.20%        | 32.40%     |
| 2                | 13.30%    | 21.65%        | 21.60%        | 61.10%     |
| 3                | 16.50%    | 16.90%        | 16.30%        | 62.95%     |
| 4                | 17.70%    | 15.45%        | 15.45%        | 67.55%     |
| 5                | 12.75%    | 14.40%        | 14.65%        | 64.95%     |
| 6                | 11.70%    | 14.10%        | 14.00%        | 65.30%     |

### Observations
1. **Performance Degradation**: In all PGL variants, adding more layers led to a decrease in test accuracy compared to the initial linear model.
2. **Loss Explosion**: In the basic PGL, the training MSE loss exploded as more layers were added, reaching extremely high values. This suggests significant numerical instability.
3. **Linearity Issue**: The "Linear Constrained" variant helped slow down the degradation but did not prevent it. This confirms the hypothesis that the mapping from inputs $X$ to optimized targets $Z$ is highly non-linear and cannot be effectively captured by a single linear layer.
4. **Fixed Top Stack**: A major limitation of this method is that the top layers are kept fixed after their initial training. Since $W_0$ was optimized for the original input distribution $X$, it becomes suboptimal when the input distribution is changed by prepending a new layer $\sigma(W X + b)$.

## Analysis
The core issue with prepending layers greedily is twofold:
1. **Unlearnable Targets**: The targets $Z$ that minimize the loss for a fixed top stack are generally not linearizable from the original inputs $X$. Fitting a linear layer to these targets results in a poor approximation.
2. **Distribution Shift**: By changing the input to the fixed top stack, we invalidate the optimization performed in previous steps. Unlike end-to-end backpropagation or standard greedy stacking (where bottom layers are fixed and top layers are trained), prepending requires the existing top stack to be robust to the transformations introduced by the new bottom layers, which is not guaranteed.

## Conclusion
Prepended greedy layer-wise learning, as implemented here, is not an effective way to train deep networks. While it uses least squares for speed, the decoupling of layers leads to poor generalization and numerical instability. Standard backpropagation remains significantly superior for this task.
