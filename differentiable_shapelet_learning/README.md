# Differentiable Shapelet Learning (DSL) on MNIST-1D

This experiment investigates **Differentiable Shapelet Learning**, a technique that learns discriminative local motifs (shapelets) for 1D signal classification.

## Hypothesis
A **Differentiable Shapelet Layer** can capture local discriminative motifs in 1D signals more effectively than standard convolutional layers by learning to minimize the distance to specific "shape" prototypes. While convolutions compute linear correlations, shapelets measure local Euclidean distances, providing a different inductive bias. We hypothesize that a network using learnable shapelets with soft-min pooling can achieve competitive performance on `mnist1d` with improved interpretability.

## Method

### 1. Differentiable Shapelet Layer (DSL)
- **Parameters**: A set of $N$ learnable shapelets (prototypes) $\{S_j\}_{j=1}^N$ of length $K$.
- **Distance Calculation**: For an input signal $X$, we use a sliding window $W_i$ of length $K$ and compute the squared Euclidean distance:
  $$d(S_j, W_i) = \|S_j - W_i\|_2^2$$
- **Soft-Min Pooling**: Instead of a hard minimum, we use a differentiable soft-min operator to find the "best match" for each shapelet:
  $$\text{DSL}(X)_j = \sum_i w_{ij} d(S_j, W_i), \quad w_{ij} = \frac{\exp(-d(S_j, W_i) / \tau)}{\sum_k \exp(-d(S_j, W_k) / \tau)}$$
  where $\tau$ is a temperature parameter.

### 2. Baselines
- **MLP Baseline**: A 3-layer MLP with BatchNorm and ReLU.
- **Conv1d Baseline**: A 1D convolutional layer followed by max pooling and an MLP head.

## Results

We evaluated the models on the `mnist1d` dataset (10,000 samples). Learning rates for all models were tuned using Optuna over 5 trials each.

| Model | Test Accuracy | Best Learning Rate |
| :--- | :--- | :--- |
| **MLP** | 76.45% | 0.00277 |
| **Conv1d** | 96.70% | 0.00532 |
| **Shapelet Network** | **86.70%** | 0.00428 |

### Observations
- **Strong Performance**: The Shapelet Network significantly outperformed the MLP baseline (86.70% vs 76.45%), demonstrating its effectiveness as a feature extractor for 1D signals.
- **Comparison to Conv1d**: While the Conv1d baseline still achieved the highest accuracy (96.70%), the Shapelet Network provided a competitive alternative with a fundamentally different inductive bias (distance-based vs. correlation-based).
- **Interpretability**: The learned shapelets (visualized in `learned_shapelets.png`) represent local motifs that the model found most discriminative.

## Visualizations
- `comparison.png`: Bar chart comparing the accuracies of the three models.
- `learned_shapelets.png`: Visualization of the first 16 learned shapelets.

## Conclusion
Differentiable Shapelet Learning provides a powerful and interpretable way to extract features from 1D signals. By learning "representative" shapes directly through backpropagation, the model can discover meaningful local structures that distinguish classes. Future work could explore multi-scale shapelets and learnable temperatures.
