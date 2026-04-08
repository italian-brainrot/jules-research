# Differentiable Tropical Networks Experiment

## Hypothesis
Tropical networks, based on the $(\max, +)$ semiring, provide a different inductive bias from standard neural networks. We hypothesize that a **Differentiable Tropical Linear Layer**, which smoothly interpolates between max, mean, and min operations via a learnable temperature parameter $\beta$, can effectively learn structured features for signal classification.

## Methodology
The `TropicalLinear` layer computes:
$$ y_j = \frac{1}{\beta_j} \left( \text{logsumexp}(\beta_j (x + W_{ij})) - \log(I) \right) $$
where $x$ is the input, $W$ is the learnable weight matrix (additive offset), $I$ is the number of input features, and $\beta_j$ is a per-neuron learnable temperature.

-   When $\beta \to \infty$, $y \to \max(x + W)$.
-   When $\beta \to -\infty$, $y \to \min(x + W)$.
-   When $\beta \to 0$, $y \to \text{mean}(x + W)$.

We evaluated three models on `mnist1d` (10k samples):
1.  **Baseline MLP**: Standard MLP with ReLU activations.
2.  **Tropical MLP**: MLP using `TropicalLinear` layers followed by `BatchNorm1d` (to handle potential scale shifts).
3.  **Tropical Augmented MLP**: Standard MLP with raw features concatenated with features from a `TropicalLinear` layer.

## Results

| Model                | Best Test Accuracy |
|----------------------|--------------------|
| Baseline MLP (4 layers)| 74.80%             |
| Tropical MLP (2 layers)| 65.90%             |
| Augmented MLP (2 layers)| 69.15%             |

### Key Observations:
1.  **Inductive Bias**: The `TropicalMLP` achieved 65.90% accuracy, demonstrating that tropical operations can indeed be used for signal classification. While it underperforms a tuned baseline, its performance is respectable given the architectural differences.
2.  **Learned Temperature**: The mean learned $\beta$ for the `TropicalMLP` shifted from the initial 1.0 to approximately 1.5, suggesting the model favored operations closer to `max` than `mean`.
3.  **Combination**: Augmenting the MLP with tropical features (69.15%) outperformed the pure `TropicalMLP`, indicating that tropical features provide complementary information but standard linear layers are still necessary for peak performance on `mnist1d`.
4.  **Stability**: Adding `BatchNorm1d` to the `TropicalMLP` significantly improved performance from ~56% to ~66%, as tropical operations do not have the same normalization properties as standard linear layers.

## Conclusion
Differentiable Tropical Networks offer a unique way to incorporate max/min-plus logic into deep learning. While they do not currently exceed the performance of standard MLPs on `mnist1d`, they provide a structured alternative for learning piecewise linear functions and could be beneficial for tasks requiring robustness to certain non-linear deformations or for specialized signal processing applications.
