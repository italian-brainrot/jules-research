# Logit Cohesion Loss (LCL) Experiment

This experiment investigates **Logit Cohesion Loss (LCL)**, a regularization technique that penalizes the variance of logit vectors within each class in a training batch.

## Hypothesis

Standard Cross-Entropy (CE) loss pushes the correct logit up and incorrect logits down, but it does not explicitly encourage consistency in the logit patterns across different samples of the same class. On a dataset like MNIST1D, which contains significant variations due to shifts and noise, forcing the model to produce similar logit vectors for all samples within a class (Logit Cohesion) should encourage the model to ignore non-discriminative variations and focus on class-essential features.

## Method

### Logit Cohesion Loss (LCL)

For a batch of samples, let $Z_i$ be the logit vector for sample $i$ and $y_i$ be its label. Let $S_c = \{i : y_i = c\}$ be the set of indices of samples belonging to class $c$. The class-wise mean logit vector is:
$$\bar{Z}_c = \frac{1}{|S_c|} \sum_{i \in S_c} Z_i$$

The Logit Cohesion Loss is defined as the average squared Euclidean distance from the class-wise mean:
$$L_{LCL} = \frac{1}{K'} \sum_{c \in \text{Classes in batch}} \frac{1}{|S_c|} \sum_{i \in S_c} ||Z_i - \bar{Z}_c||^2$$
where $K'$ is the number of classes present in the batch with at least two samples.

The total loss is:
$$L = L_{CE} + \lambda \cdot L_{LCL}$$

## Experimental Setup

- **Dataset:** MNIST1D (10,000 samples)
- **Model:** 3-layer MLP (40 -> 256 -> 256 -> 10)
- **Optimizer:** Adam
- **Hyperparameter Tuning:** Optuna (30 trials each)
- **Baselines:**
  - Standard Cross-Entropy (CE)
  - Label Smoothing (CE + LS)

## Results

| Method | Best Hyperparameters | Test Accuracy |
|--------|----------------------|---------------|
| Cross-Entropy | LR: 0.0047 | 75.85% |
| Label Smoothing | LR: 0.0052, $\epsilon$: 0.224 | 75.60% |
| **Logit Cohesion (Ours)** | **LR: 0.0042, $\lambda$: 0.0175** | **77.15%** |

## Analysis

- **Performance:** Logit Cohesion Loss achieved a test accuracy of **77.15%**, which is a **1.3% absolute improvement** over the standard Cross-Entropy baseline.
- **Comparison to Label Smoothing:** Label Smoothing slightly underperformed standard CE in this specific setup, whereas LCL provided a consistent boost.
- **Why it works:** LCL acts as a strong regularizer in the logit space. By forcing all samples of a class to map to the same logit vector, it prevents the model from over-optimizing the prediction for "easy" samples (e.g., by pushing the correct logit to extremely large values) at the expense of "hard" samples. It encourages a more compact and consistent representation of classes in the decision space.

## Conclusion

Logit Cohesion Loss is a simple yet effective regularization technique for 1D signal classification. It is computationally efficient as it only requires intra-batch calculations and does not need extra forward passes or complex memory structures (unlike Center Loss which requires maintaining global centroids).
