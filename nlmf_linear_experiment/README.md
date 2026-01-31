# Non-Linear Matrix Factorization (NLMF) Experiment

This experiment investigates whether applying a non-linear activation (via a small element-wise MLP) to a low-rank matrix factorization can improve its performance as a parameter-efficient weight matrix representation.

## Hypothesis
Standard low-rank approximations ($W = U V^T$) are restricted to matrices on the low-rank manifold. By applying a non-linear function $\phi$ element-wise, $W = \phi(U V^T)$, we can potentially represent higher-rank structures while still only learning a small number of parameters (the factors $U, V$ and the parameters of $\phi$). This might allow the network to capture more complex patterns than a pure low-rank layer with a similar parameter budget.

## Methodology
- **Dataset**: MNIST-1D (10,000 samples).
- **Architecture**: A 2-hidden-layer MLP (Hidden size 100).
- **Layers Compared**:
    - **Dense**: Standard `nn.Linear` (5110 parameters).
    - **Low-Rank**: $W = U V^T$ with rank 8 (~2110 parameters).
    - **NLMF**: $W = \phi(U V^T)$ with rank 4 and a small 1x1 Conv-based MLP (1-16-1 architecture) (~1208 parameters).
    - **Kronecker**: $W = A \otimes B$ (~320 parameters).
- **Hyperparameter Tuning**: Optuna was used to tune the learning rate for each model independently (15 trials each).
- **Training**: Each model was trained for 50 epochs using the best found learning rate with the Adam optimizer.

## Results

| Model | Parameters | Best Learning Rate | Final Test Accuracy |
|-------|------------|--------------------|---------------------|
| Dense | 5110 | 0.0253 | 70.65% |
| Low-Rank (r=8) | 2110 | 0.0108 | 59.55% |
| NLMF (r=4) | 1208 | 0.0689 | 46.35% |
| Kronecker | 320 | 0.0964 | 53.00% |

The convergence plot is available in `comparison_results.png`.

## Analysis
- **NLMF Performance**: Surprisingly, NLMF performed significantly worse than both the Dense baseline and the Low-Rank baseline. More importantly, it was even outperformed by the Kronecker product model, which used roughly 4x fewer parameters.
- **Complexity vs. Trainability**: The non-linear transformation $\phi(U V^T)$ might make the optimization landscape much more difficult for Adam to navigate. The high optimal learning rate for NLMF (0.0689) suggests that the gradients might be small or unstable.
- **Structural Bias**: Low-rank and Kronecker structures provide a strong inductive bias that seems beneficial for this task. NLMF's non-linearity might be "too flexible" or "too destructive" to the useful structural properties of the low-rank factorization.

## Conclusion
The hypothesis that non-linear matrix factorization would outperform standard low-rank approximations was not supported by this experiment on MNIST-1D. While NLMF can theoretically represent higher-rank matrices, the added complexity of the element-wise MLP appears to hinder training rather than help it. For parameter-efficient architectures on this dataset, structured weight matrices like Kronecker products remain superior.
