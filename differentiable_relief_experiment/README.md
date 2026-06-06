# Differentiable Relief-inspired Loss Experiment

## Hypothesis
Differentiable Relief-inspired Loss (DRelief) improves representation learning by encouraging local neighborhood consistency. By penalizing features where same-class neighbors (hits) are farther than different-class neighbors (misses), the model learns more discriminative and well-clustered embeddings.

## Methodology
- **Loss Function**: `DReliefLoss` implements a soft version of the ReliefF algorithm.
  - It computes pairwise distances in the feature space.
  - It uses a softmin-based approach (via `F.softmax` on negative distances) to identify "hits" (nearest samples of the same class) and "misses" (nearest samples of each other class).
  - The objective is to minimize `HitDist - MissDist`.
- **Dataset**: `mnist1d` (10,000 samples).
- **Model**: A 3-layer MLP with BatchNorm and ReLU.
- **Comparison**:
  - **Baseline**: Standard MLP trained with Cross-Entropy loss.
  - **Relief**: Same MLP trained with Cross-Entropy plus an auxiliary DRelief loss applied to the penultimate layer's features.
- **Fair Comparison**: Both models were tuned using Optuna (10 trials each) for learning rate and weight decay. The Relief model also had its `lambda_relief` and `temperature` tuned.
- **Evaluation**: 5 seeds, 40 epochs per seed.

## Results
The experiment yielded the following results on the `mnist1d` test set:

| Method | Mean Test Accuracy | Std Dev |
| :--- | :---: | :---: |
| **Baseline** | **77.09%** | 0.69% |
| **Relief** | 76.16% | 0.53% |

## Observations
- The baseline MLP performed slightly better than the Relief-regularized model.
- The tuned `lambda_relief` was quite small (approx 0.00015), suggesting that the optimization preferred a very weak regularization from the Relief loss.
- In this specific setup, the raw signal features and standard cross-entropy were sufficient for the task, and the additional local clustering constraint did not provide a performance boost.

## Conclusion
While the Differentiable Relief loss correctly implements the intuition of the Relief algorithm in a gradient-based framework, it did not improve performance on the `mnist1d` dataset. This might be due to the dataset already having well-separated classes or the auxiliary loss interfering with the primary classification objective.
