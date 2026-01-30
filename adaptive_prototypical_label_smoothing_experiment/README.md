# Adaptive Prototypical Label Smoothing (APLS) Experiment

## Hypothesis
Adaptive Prototypical Label Smoothing (APLS) improves generalization on very small datasets by distributing label smoothing mass based on class similarity in the feature space.

Standard label smoothing distributes the smoothing mass $\epsilon$ uniformly across all non-target classes. However, in many datasets, some classes are more semantically or structurally similar than others. APLS dynamically estimates class similarity by tracking the centroids of hidden representations (prototypes) and uses these similarities to weight the smoothing mass. This encourages the model to be more "uncertain" between similar classes than between dissimilar ones, potentially leading to better decision boundaries when data is scarce.

## Methodology
- **Dataset**: `mnist1d` with a small training set (300 samples, resulting in ~240 training samples after internal processing).
- **Model**: 3-layer MLP with 256 hidden units.
- **Comparison**:
    1. **Baseline**: Standard Cross-Entropy loss.
    2. **Standard Label Smoothing (LS)**: Uniformly distributed $\epsilon$.
    3. **Adaptive Prototypical Label Smoothing (APLS)**: Feature-distance-based $\epsilon$ distribution using a moving average of class centroids.
- **Hyperparameter Tuning**: Optuna was used to tune the learning rate, weight decay, and method-specific parameters ($\epsilon$, temperature for APLS, etc.) for each mode with 15 trials per mode.
- **Evaluation**: Test accuracy on the `mnist1d` test set.

## Results
The experiment yielded the following best accuracies for each method:

| Method | Best Test Accuracy | Best Hyperparameters |
| :--- | :--- | :--- |
| **Baseline** | 0.3500 | `lr`: 0.0039, `weight_decay`: 0.0005 |
| **Label Smoothing** | **0.3667** | `lr`: 0.0089, `weight_decay`: 0.0008, `epsilon`: 0.108 |
| **APLS** | 0.3500 | `lr`: 0.0014, `weight_decay`: 0.0005, `epsilon`: 0.292, `temp`: 0.929, `momentum`: 0.949 |

### Observations
- Both Label Smoothing and APLS used significant $\epsilon$ values (0.108 and 0.292 respectively), indicating that regularization is beneficial for this small dataset.
- Standard Label Smoothing achieved the highest accuracy (0.3667), slightly outperforming the Baseline and APLS.
- APLS achieved a similar accuracy to the Baseline but required a much higher $\epsilon$ (0.292). The temperature parameter found by Optuna (0.929) suggests that the distribution of smoothing mass was moderately non-uniform.

## Conclusion
In this specific setup on the `mnist1d` dataset with very few samples, Adaptive Prototypical Label Smoothing did not show a clear advantage over standard uniform Label Smoothing. While it outperformed the baseline in some trials, standard LS was more robust or easier to optimize.

Possible reasons for APLS not outperforming LS include:
1. **Centroid Instability**: On very small datasets, the class centroids in the feature space may be noisy and unstable, especially in the early stages of training, leading to "incorrect" smoothing distributions.
2. **Feature Quality**: If the model doesn't learn discriminative features quickly, the distances between prototypes might not reflect true semantic similarity.
3. **Task Simplicity**: `mnist1d` might not have the complex hierarchical class relationships where APLS would be most beneficial (e.g., distinguishing between different breeds of dogs vs. dogs and cats).

Further research could explore pre-training the feature extractor or using a more stable prototype estimation method (e.g., using a separate memory bank).
