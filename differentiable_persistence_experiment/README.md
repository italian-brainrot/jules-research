# Differentiable 1D Persistence Experiment

This experiment evaluates a novel **Differentiable 1D Persistence Layer** for signal classification.

## Hypothesis

Persistent Homology is a tool from Topological Data Analysis (TDA) that captures the structural features of a signal (e.g., prominent peaks and valleys) that persist across different scales. We hypothesize that by making 0D persistence (birth-death pairs of sublevel and superlevel sets) differentiable, a neural network can leverage these topological features to improve classification performance or learn more robust representations.

## Methodology

### 1. Differentiable 1D Persistence Layer
- **Process**: Computes the 0-dimensional persistence of a 1D signal. It identifies local minima (births) and their corresponding deaths at local maxima where they merge with a deeper component (Elder Rule).
- **Features**: The top-$k$ persistence values (death - birth) are extracted for both sublevel set filtration (original signal) and superlevel set filtration (negated signal).
- **Differentiability**: The layer is implemented using vectorized PyTorch operations. Since the indices of extrema are locally constant, the persistence values are differentiable with respect to the input signal values at those indices.

### 2. Experimental Setup
- **Dataset**: `mnist1d` (10,000 samples).
- **Models**:
  - `BaselineMLP`: Standard 3-layer MLP.
  - `PersistenceAugmentedMLP`: Baseline MLP augmented with top-10 sublevel and top-10 superlevel persistence features.
  - `PersistenceMLP`: MLP using *only* the 20 persistence features.
- **Tuning**: Learning rates were tuned using Optuna for 10 trials each.
- **Evaluation**: Final accuracy was averaged over 3 different seeds.

## Results

| Model | Test Accuracy (Mean +/- Std) | Best LR |
|---|---|---|
| Baseline MLP | 73.58% +/- 0.89% | 0.00618 |
| Persistence-Augmented MLP | 71.32% +/- 0.25% | 0.00114 |
| Persistence-Only MLP | 46.85% +/- 0.55% | 0.00538 |

## Conclusion

The experiment shows that while the **Differentiable 1D Persistence Layer** extracts discriminative features (achieving ~47% accuracy on its own compared to 10% random chance), it did not provide an additive benefit to the baseline MLP on the MNIST-1D dataset. In fact, the augmented model performed slightly worse, suggesting that the raw signal already contains the necessary topological information in a form that the MLP can easily extract, or that the additional features introduced noise/overfitting.

However, the success of the `Persistence-Only` model confirms that topological persistence captures significant structural information about the signal.

## Visualizations

Comparison of results is available in `comparison.png`.
