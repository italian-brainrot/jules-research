# Arnoldi Feature Layer Experiment

This experiment evaluates a novel **Differentiable Arnoldi Feature Layer** for signal and tabular classification.

## Hypothesis

The Arnoldi process is a classical linear algebra algorithm used to compute an orthonormal basis for a Krylov subspace $\mathcal{K}_k(A, x) = \text{span}\{x, Ax, A^2x, \dots, A^{k-1}x\}$. By treating the input feature vector $x$ as the starting vector and the matrix $A$ as a learnable parameter, the Arnoldi process can extract structured, high-order features (the upper Hessenberg matrix $H$).

We hypothesize that these features provide a useful inductive bias for capturing non-linear interactions between input features and learned transformations, potentially outperforming standard dense layers.

## Methodology

### 1. Arnoldi Feature Layer
- **Process**: Performs $k$ steps of the Arnoldi process on input $x \in \mathbb{R}^d$ and a learnable matrix $A \in \mathbb{R}^{d \times d}$.
- **Output**: The upper Hessenberg matrix $H \in \mathbb{R}^{(k+1) \times k}$, which is flattened and used as a feature vector.
- **Differentiability**: The process is implemented using differentiable PyTorch operations (norms, inner products, and matrix-vector multiplications), allowing $A$ to be learned via backpropagation.

### 2. Experimental Setup
- **Dataset**: `mnist1d` (10,000 samples).
- **Architecture (Arnoldi-MLP)**: Input $x$ is concatenated with Arnoldi features from 2 heads (each with $k=3$). This augmented vector is passed through a 3-layer MLP.
- **Architecture (Baseline)**: A standard 3-layer MLP with 128 hidden units.
- **Tuning**: Learning rates for both models were tuned using Optuna for 5 trials (15 epochs each).
- **Evaluation**: Final evaluation over 40 epochs with 2 different seeds.

## Results

| Model | Test Accuracy (Mean +/- Std) |
|---|---|
| Baseline MLP | 74.58% +/- 0.23% |
| **Arnoldi-MLP** | **77.02% +/- 0.28%** |

The Arnoldi-MLP consistently outperformed the baseline MLP on the MNIST-1D task.

## Conclusion

The Differentiable Arnoldi Feature Layer provides a significant performance boost over a standard MLP by extracting structured Krylov subspace features. This indicates that incorporating classical linear algebra algorithms as differentiable layers can introduce useful inductive biases for signal-like data.

## Visualizations

The training progress can be seen in `comparison.png`.
