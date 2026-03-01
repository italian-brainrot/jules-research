# Neuron-wise Gradient Standardization (NGS) Experiment

## Hypothesis
We hypothesize that **Neuron-wise Gradient Standardization (NGS)**, which extends **Gradient Centralization (GC)** by not only centering but also standardizing the variance of gradients across the input weights of each neuron, can improve optimization stability and generalization.

Standard GC centers the gradients of each output neuron's weights (rows of the weight matrix for a Linear layer). NGS further scales these gradients so that each neuron receives an update of the same magnitude (unit variance). This can be seen as a form of "internal covariate shift" correction for gradients, ensuring that no single neuron's update dominates the learning process due to gradient scale imbalances.

## Methodology
- **Dataset**: `mnist1d` (4,000 samples).
- **Model**: 3-layer MLP (40 -> 256 -> 256 -> 10).
- **Optimizer**: Adam.
- **Modes**:
  - **Baseline**: Standard Adam.
  - **GC (Gradient Centralization)**: Centering gradients per neuron: $g_{i, \cdot} \leftarrow g_{i, \cdot} - \text{mean}(g_{i, \cdot})$.
  - **NGS (Neuron-wise Gradient Standardization)**: Centering and scaling gradients per neuron: $g_{i, \cdot} \leftarrow \frac{g_{i, \cdot} - \text{mean}(g_{i, \cdot})}{\text{std}(g_{i, \cdot}) + \epsilon}$.
- **Tuning**: Optuna was used to tune the learning rate and weight decay for each mode (20 trials each, 20 epochs per trial).
- **Evaluation**: Final evaluation over 5 different random seeds for 100 epochs using the best hyperparameters.

## Results

| Mode | Mean Val Accuracy | Mean Test Accuracy | Best Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Baseline** | **0.6466 ± 0.0081** | **0.6368 ± 0.0054** | `lr`: 8.53e-3, `wd`: 4.24e-4 |
| **GC** | 0.6069 ± 0.0046 | 0.5985 ± 0.0076 | `lr`: 4.14e-3, `wd`: 1.18e-5 |
| **NGS** | 0.6000 ± 0.0075 | 0.5797 ± 0.0164 | `lr`: 8.84e-3, `wd`: 6.99e-5 |

### Analysis
- **Baseline significantly outperformed both GC and NGS.**
- Both GC and NGS actually hindered the performance on this specific task.
- NGS performed the worst, especially on the test set, showing higher variance across seeds (±0.0164).
- The training curves (see `results.png`) showed that while NGS and GC converged, they did not reach the same level of accuracy as the baseline Adam optimizer.

## Conclusion
The hypothesis that Neuron-wise Gradient Standardization (NGS) would improve performance was **not supported** by the results on `mnist1d`. It appears that for this dataset and architecture, the relative scales of gradients across neurons carry important information that is lost or distorted when standardization is applied. Gradient Centralization also performed worse than the baseline, suggesting that even the mean gradient per neuron is a useful signal that shouldn't be discarded in this context.

Future work could investigate whether these techniques are more beneficial for much deeper networks or different architectures (like CNNs where GC was originally proposed) where gradient vanishing/exploding or scale imbalances are more prevalent.

## Visualizations
The training loss and validation accuracy curves for all modes are available in `results.png`.
