# Differentiable Soft Decision Tree Experiment

This experiment explores the integration of a Differentiable Soft Decision Tree (SDT) as a feature extractor or augmentation for a standard Multi-Layer Perceptron (MLP) on the MNIST-1D signal classification task.

## Hypothesis
Soft Decision Trees provide a hierarchical, interpretable routing mechanism. By making them differentiable, we allow them to be trained via backpropagation. We hypothesize that augmenting a standard MLP with features from a Soft Decision Tree can provide a beneficial inductive bias, potentially capturing hierarchical structures in the input signals that standard flat MLPs might struggle with.

## Model Architecture

### Soft Decision Tree (SDT)
The SDT consists of:
- **Inner Nodes**: Each node $i$ has a learnable weight vector $w_i$ and bias $b_i$. It computes a routing probability $g_i(x) = \sigma(\beta(w_i \cdot x + b_i))$, where $\sigma$ is the sigmoid function and $\beta$ is a temperature parameter.
- **Path Probabilities**: The probability of reaching a node is the product of the gating probabilities along the path from the root.
- **Leaf Nodes**: Each leaf node $j$ has a learnable parameter vector $\theta_j$.
- **Output**: The final output is the expected prediction over all leaf nodes: $y = \sum_{j \in \text{leaves}} p_j(x) \theta_j$.

### Models Compared
1. **BaselineMLP**: A 3-layer MLP with ReLU activations.
2. **SoftDecisionTree**: A standalone SDT classifier (depth 6).
3. **SDTAugmentedMLP**: An MLP where the input to the final fully connected layer is the sum of a standard MLP's hidden features and an SDT's output (SDT depth 5).

## Experimental Setup
- **Dataset**: MNIST-1D (10,000 samples).
- **Tuning**: Learning rates were tuned for each model using Optuna (10 trials each) on a subset of the training data.
- **Evaluation**: Each model was trained and evaluated over 3 different seeds for 30 epochs.

## Results

| Model | Accuracy (Mean ± Std) |
| :--- | :--- |
| BaselineMLP | 0.7652 ± 0.0100 |
| SoftDecisionTree | 0.5140 ± 0.0126 |
| **SDTAugmentedMLP** | **0.7945 ± 0.0123** |

The SDTAugmentedMLP outperformed the BaselineMLP by approximately 3%, suggesting that the hierarchical routing of the Soft Decision Tree provides useful complementary features to the standard MLP architecture. The standalone SDT performed significantly worse, likely due to its limited representational capacity compared to the deeper MLP.

## Visualizations
The results are summarized in `results.png`.
