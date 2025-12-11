# Eigen-Conjugate Gradient Descent Experiment

## Hypothesis

The convergence of the Conjugate Gradient (CG) algorithm can be accelerated by incorporating information from the Hessian's eigenspectrum. Specifically, by periodically projecting the gradient updates onto the subspace spanned by the top eigenvectors of the Hessian, the optimizer can take more effective steps along the directions of highest curvature, leading to faster convergence.

## Methodology

### Optimizers

- **Standard Conjugate Gradient (CG):** A baseline implementation using Hessian-vector products.
- **Eigen-Conjugate Gradient (Eigen-CG):** The proposed optimizer. It's a modified CG that periodically computes the top 5 eigenvectors of the Hessian using the Lanczos method (from the `pytorch-hessian-eigenthings` library) and projects the CG search direction onto the subspace spanned by these eigenvectors. The eigenvector update frequency was set to every 10 steps.

### Model and Dataset

- **Model:** A simple Convolutional Neural Network (CNN) with two convolutional layers and two fully-connected layers.
- **Dataset:** A subset of the MNIST dataset (6400 training images) to ensure the experiment completes within a reasonable time.
- **Training:** Both models were trained for 1 epoch with a learning rate of 0.01.

## Results

The experiment was run, and the output was saved to `experiment_results.txt`. Here's a summary of the results:

| Optimizer | Test Loss | Test Accuracy |
|---|---|---|
| Standard CG | 2.3430 | 12% |
| Eigen-CG | 2.3609 | 1% |

The standard Conjugate Gradient optimizer performed significantly better than the Eigen-Conjugate Gradient optimizer.

## Conclusion

The hypothesis was not supported by the results of this experiment. The Eigen-Conjugate Gradient optimizer performed significantly worse than the standard CG optimizer. There are several potential reasons for this:

1.  **High Computational Cost:** The periodic eigenvector calculation is computationally expensive, which slows down the training process significantly.
2.  **Inaccurate Eigenvector Estimation:** The eigenvectors are estimated on a single batch of data at each step where they are computed. This estimation might be too noisy to provide a useful direction for the optimizer.
3.  **Projection Issues:** Projecting the search direction onto a low-dimensional subspace might be too restrictive, preventing the optimizer from exploring other important directions in the loss landscape.
4.  **Lack of Hyperparameter Tuning:** The learning rate, number of eigenvectors, and update frequency were not tuned. It's possible that a different set of hyperparameters could lead to better performance.

Further research could explore more efficient methods for estimating the top eigenvectors or using them in a less restrictive way to guide the search direction.
