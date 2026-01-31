import torch
import numpy as np
from rational_feature_expansion_experiment.model import RationalFeatureExpansion

def test_rational_regression():
    # Toy problem: y = (2x + 1) / (x + 2)
    X = torch.linspace(-1, 1, 100).view(-1, 1)
    y = (2*X + 1) / (X + 2)

    # RFE should be able to fit this exactly if we use x as the only feature
    model = RationalFeatureExpansion(num_features=2, num_iter=10, reg=1e-8)
    def custom_get_rff(x):
        ones = torch.ones(x.shape[0], 1)
        return torch.cat([x, ones], dim=-1)

    model._get_rff = custom_get_rff

    # fit expects y as (N, C)
    model.fit(X, y.view(-1, 1))

    preds = model.predict_proba(X)
    mse = torch.mean((preds - y.view(-1, 1))**2).item()
    print(f"MSE: {mse}")
    assert mse < 1e-4

if __name__ == "__main__":
    test_rational_regression()
