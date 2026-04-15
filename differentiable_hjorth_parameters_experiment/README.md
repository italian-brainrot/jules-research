# Differentiable Hjorth Parameters Experiment

This experiment evaluates the effectiveness of Hjorth parameters (Activity, Mobility, and Complexity) as features for signal classification using the `mnist1d` dataset.

## Hjorth Parameters

Hjorth parameters are time-domain descriptors of EEG signals, but they can be applied to any 1D signal:

1.  **Activity**: The variance of the signal amplitude. $Activity = \text{var}(y(t))$
2.  **Mobility**: Represents the mean frequency or the proportion of standard deviation of the power spectrum. $Mobility = \sqrt{\frac{\text{var}(y'(t))}{\text{var}(y(t))}}$
3.  **Complexity**: Indicates the similarity of the shape of the signal to a pure sine wave. $Complexity = \frac{Mobility(y'(t))}{Mobility(y(t))}$

We implemented these parameters as a differentiable PyTorch layer, allowing them to be used as part of a neural network.

## Experimental Setup

We compared three models:
1.  **BaselineMLP**: A standard 3-layer MLP processing the raw 1D signal.
2.  **HjorthMLP**: An MLP processing *only* the 3 Hjorth parameters.
3.  **HjorthAugmentedMLP**: An MLP processing the raw 1D signal concatenated with the 3 Hjorth parameters.

All models were tuned for learning rate using Optuna (20 trials) and then evaluated across 5 seeds.

## Results

| Model | Accuracy (%) | Best LR |
| :--- | :--- | :--- |
| BaselineMLP | 73.93 +/- 0.96 | 0.0090 |
| HjorthMLP | 29.78 +/- 0.57 | 0.0100 |
| HjorthAugmentedMLP | **75.55 +/- 0.62** | 0.0074 |

## Conclusion

The `HjorthAugmentedMLP` slightly outperformed the `BaselineMLP`, suggesting that Hjorth parameters provide useful inductive biases that are not entirely captured or easily learned by a standard MLP from raw data. However, using Hjorth parameters alone (`HjorthMLP`) resulted in poor performance, which is expected as they represent a significant compression of the signal (from 40 dimensions down to 3).

The differentiability of the Hjorth parameters allows them to be used in any end-to-end trainable architecture, potentially providing a simple but effective feature engineering boost for time-series tasks.
