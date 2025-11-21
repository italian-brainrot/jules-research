import numpy as np
from sklearn.datasets import make_classification

# Generate a base dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    n_classes=2,
    random_state=42,
)

# Introduce multicollinearity
n_collinear_features = 5
for _ in range(n_collinear_features):
    # Select two random features
    feature1_idx, feature2_idx = np.random.choice(X.shape[1], 2, replace=False)

    # Create a new feature that is a linear combination of the two selected features
    new_feature = 0.5 * X[:, feature1_idx] + 0.5 * X[:, feature2_idx]

    # Add some noise
    noise = np.random.normal(0, 0.1, X.shape[0])
    new_feature += noise

    # Add the new feature to the dataset
    X = np.c_[X, new_feature]

# Save the dataset
output_path = 'gradient_compression_optimizer/logistic_regression_dataset.npz'
np.savez(output_path, X=X, y=y)

print(f"Dataset generated and saved to '{output_path}'")
