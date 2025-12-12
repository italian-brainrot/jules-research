# Eigen-Conjugate Gradient Descent Experiment

## Hypothesis

The convergence of the Conjugate Gradient (CG) algorithm can be accelerated by incorporating information from the Hessian's eigenspectrum. Specifically, by periodically projecting the gradient updates onto the subspace spanned by the top eigenvectors of the Hessian, the optimizer can take more effective steps along the directions of highest curvature, leading to faster convergence.

## Methodology

### Optimizers

- **Standard Conjugate Gradient (CG):** A baseline implementation using Hessian-vector products.
- **Eigen-Conjugate Gradient (Eigen-CG):** The proposed optimizer. It's a modified CG that periodically computes the top 5 eigenvectors of the Hessian using the Lanczos method (from the `pytorch-hessian-eigenthings` library) and projects the CG search direction onto the subspace spanned by these eigenvectors. The eigenvector update frequency was set to every 10 steps.

### Model and Dataset

- **Model:** A simple Multi-Layer Perceptron (MLP) with one hidden layer.
- **Dataset:** The `mnist1d` dataset, which is a 1D version of the MNIST dataset.
- **Training:** Both models were trained for 1 epoch. The learning rate for each optimizer was tuned by selecting the best performing learning rate from `[0.001, 0.01, 0.1]`.

## Results

The experiment was run, and the output was saved to `experiment_results.txt`. Here's a summary of the results after learning rate tuning:

| Optimizer | Best Learning Rate | Test Loss | Test Accuracy |
|---|---|---|---|
| Standard CG | 0.1 | 2.3064 | 9% |
| Eigen-CG | 0.1 | 2.3075 | 9% |

After tuning the learning rate, both optimizers performed similarly, with the standard Conjugate Gradient optimizer achieving a slightly lower loss.

## Conclusion

The hypothesis was not supported by the results of this experiment. Even with learning rate tuning and a simpler dataset, the Eigen-Conjugate Gradient optimizer did not outperform the standard CG optimizer. The potential reasons for this are the same as in the original experiment:

1.  **High Computational Cost:** The periodic eigenvector calculation is computationally expensive, which slows down the training process significantly.
2.  **Inaccurate Eigenvector Estimation:** The eigenvectors are estimated on a single batch of data at each step where they are computed. This estimation might be too noisy to provide a useful direction for the optimizer.
3.  **Projection Issues:** Projecting the search direction onto a low-dimensional subspace might be too restrictive, preventing the optimizer from exploring other important directions in the loss landscape.

Further research could explore more efficient methods for estimating the top eigenvectors or using them in a less restrictive way to guide the search direction.
