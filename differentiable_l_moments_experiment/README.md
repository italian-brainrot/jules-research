# Differentiable L-moments Experiment

This experiment introduces a differentiable layer for computing **L-moments**, which are alternative statistics to traditional moments (mean, variance, skewness, kurtosis) based on linear combinations of order statistics.

## Method

L-moments are robust to outliers and often provide more reliable estimates of distribution shape than conventional moments. For a signal $x$ of length $n$, the $r$-th L-moment is defined as:
$$L_r = \sum_{k=0}^{r-1} (-1)^{r-1-k} \binom{r-1}{k} \binom{r-1+k}{k} b_k$$
where $b_k$ are Probability Weighted Moments (PWMs):
$$b_k = \frac{1}{n} \sum_{j=1}^n \frac{\binom{j-1}{k}}{\binom{n-1}{k}} x_{(j)}$$
and $x_{(j)}$ are the sorted values of the signal.

In this implementation:
1. We use `torch.sort` to obtain $x_{(j)}$.
2. PWM weights are precomputed for efficiency.
3. The L-moments are computed via a matrix multiplication of PWMs and precomputed coefficients.
4. The operation is fully differentiable because `torch.sort` is differentiable (gradients flow to the values at their original positions).

We use a sliding window approach to extract local L-moment features from the 1D signal.

## Dataset
- **MNIST-1D**: A 1D version of MNIST with 10,000 samples and sequence length 40.

## Results

We compared a `BaselineMLP` against an `LMomentAugmentedMLP` which receives both the raw signal and its local L-moment features (first 4 L-moments, window size 10, stride 5). Learning rates were tuned using Optuna (12 trials).

| Model | Test Accuracy |
|-------|---------------|
| **Baseline MLP** | 79.16% ± 0.45% |
| **L-Moment Augmented MLP** | **81.55% ± 0.35%** |

The L-moment features provided a clear boost in classification performance, suggesting that robust shape descriptors are beneficial for signal classification tasks even when the raw signal is also available to the model.

## Implementation Details
- `model.py`: Contains `LMoments1d`, `LMomentSlidingWindow1d`, and model architectures.
- `train.py`: Training and tuning script using Optuna.
- `test_logic.py`: Verification of L-moment calculations and gradients.
