# Tabular Classifier Experiment: Prototype Classifier

## Hypothesis
A Prototype Classifier, which classifies samples based on their distance to a set of learnable class-specific prototypes, can effectively model tabular data by learning a non-linear distance metric. This approach is hypothesized to be more interpretable and potentially more robust than a standard MLP for certain types of tabular data.

## Methodology
The experiment compared two architectures on the `mnist1d` dataset (treated as tabular data with 40 features):
1.  **MLP (Baseline)**: A standard Multi-Layer Perceptron with ReLU activations and tuned hyperparameters (hidden dimension, number of layers, learning rate).
2.  **Prototype Classifier**: A model that learns $K$ prototypes per class. For each class, the distance to the input is calculated as the minimum (via softmin) distance to any of its prototypes. The distance metric is a learnable diagonal scaling of the Euclidean distance for each prototype.

Initially, a **Soft Oblivious Decision Forest (SODF)** was also explored, but it demonstrated poor performance (~30% accuracy) compared to the MLP baseline, likely due to optimization difficulties and limited expressive capacity of oblivious splits on this dataset.

## Results
Both models were tuned using Optuna for 10 trials each.

| Model | Best Validation Accuracy |
|-------|--------------------------|
| MLP (Baseline) | 81.95% |
| Prototype Classifier | 69.90% |

The Prototype Classifier achieved a respectable ~70% accuracy, demonstrating that it can learn to distinguish classes in this high-dimensional space, although it did not outperform the highly flexible MLP baseline in this specific configuration.

## Conclusion
While the Prototype Classifier did not surpass the MLP baseline on the `mnist1d` dataset, it showed significant improvement over the initial soft decision tree attempts. The learned prototypes and distance metrics provide a structured way to represent class regions in the feature space. Further improvements could include using more complex distance metrics (e.g., Mahalanobis distance with low-rank covariance matrices) or hierarchical prototype structures.
