# Differentiable Piecewise Linear Representation (DPLR) Experiment

This experiment evaluates a Differentiable Piecewise Linear Representation (DPLR) layer as a feature extractor for signal classification on the MNIST-1D dataset.

## Method

DPLR decomposes a 1D signal into $K$ segments, each modeled by a linear fit ($y = at + c$).

### Differentiable Layer Implementation
1.  **Soft Segmentation**: Instead of hard boundaries, we use a "soft" boxcar function for each segment $j$, defined as the difference of two sigmoids:
    $w_{ij} = \sigma(\beta(t_i - s_j)) - \sigma(\beta(t_i - e_j))$
    where $s_j$ and $e_j$ are the start and end breakpoints of segment $j$, and $\beta$ is a temperature parameter (default 10.0). Breakpoints are learnable parameters.
2.  **Weighted Least Squares (WLS)**: For each segment, we analytically solve for the optimal slope $a_j$ and intercept $c_j$ that minimize the weighted squared error $\sum_i w_{ij}(x_i - (a_j t_i + c_j))^2$.
3.  **Feature Extraction**: The layer outputs the slope $a_j$, intercept $c_j$, and the weighted Mean Squared Error (MSE) for each segment. For $K=5$, this results in $5 \times 3 = 15$ features.

## Results

Hyperparameters (learning rate) were tuned using Optuna (10 trials). Final evaluation was performed over 3 seeds.

| Model | Mean Test Accuracy | Best LR |
|-------|--------------------|---------|
| Baseline MLP | 76.63% ± 0.12% | 0.004340 |
| DPLR MLP (Features Only) | 56.82% ± 0.28% | 0.014530 |
| DPLR Augmented MLP | **78.82% ± 1.21%** | 0.009695 |

### Analysis
- **DPLR Augmented MLP** outperformed the baseline by approximately 2.2%, suggesting that the structural features (slopes and intercepts) captured by the piecewise linear approximation provide complementary information to the raw signal.
- **DPLR MLP** alone achieved a respectable 56.82% accuracy despite using only 15 features to represent a 40-dimensional signal. This demonstrates the high information density of the piecewise linear representation.
- The differentiability of breakpoints allows the model to adaptively segment the signal to minimize classification loss, effectively learning a task-specific segmentation.

## Files
- `model.py`: Implementation of the `DPLRLayer` and MLP architectures.
- `train.py`: Training script with Optuna tuning and evaluation.
- `test_logic.py`: Logic and gradient verification for the DPLR layer.
- `results.png`: Comparison plot of the model accuracies.
