# Saddle-Free Newton on Bilinear Sine Gated Networks (BSGN)

This experiment explores the application of **Saddle-Free Newton (SFN)** optimization on a novel nonlinear architecture called the **Bisine Network**.

## Bisine Network

The Bisine Network is a sum of interaction units where each unit is a product of two sine-modulated hyperplanes:
$$z_c = \sum_{k=1}^K a_{ck} \sin(w_{ck1}^T x + b_{ck1}) \sin(w_{ck2}^T x + b_{ck2})$$
where $z_c$ is the logit for class $c$.

### Advantages
- **Exact Hessian**: The model's structure allows for the derivation of exact Gradient and Hessian expressions. Specifically, the second-order derivative of the model output with respect to its parameters is block-diagonal with respect to the units $k$.
- **Cheap Computation**: For small $K$ and dimension $D$, the full Hessian of the Cross-Entropy loss can be computed in $O(C^2 N P_c)$ where $P_c$ is the number of parameters per class.

## Saddle-Free Newton (SFN)

SFN is a second-order optimization method designed to handle saddle points in non-convex optimization. It modifies the standard Newton step by taking the absolute value of the Hessian eigenvalues:
1. Compute Hessian $H = V \Lambda V^T$.
2. Define $|H| = V |\Lambda| V^T$.
3. Update $\Delta \theta = - (|H| + \lambda I)^{-1} g$.

This ensures the step is always in a descent direction (assuming $\lambda$ is sufficient) and moves *away* from saddle points in all directions of curvature.

## Experiment Results

We compared Bisine + SFN against Bisine + Adam and a standard MLP + Adam on the `mnist1d` dataset.

| Method | Test Accuracy | Training Behavior |
| --- | --- | --- |
| Bisine + SFN | 26.5% | Extremely fast convergence to 100% train accuracy; high overfitting. |
| Bisine + Adam | 43.8% | Slower convergence but better generalization. |
| MLP + Adam | 50.5% | Strongest baseline. |

### Observations
- **Fast Convergence**: SFN reaches near-zero training loss within 40-50 steps (full batch), whereas Adam takes hundreds of epochs.
- **Saddle Points**: We observed significant negative eigenvalues in the early stages of training (steps 0-10), which SFN successfully navigated.
- **Overfitting**: The exactness of the second-order information leads the model to perfectly fit the training data (including noise), resulting in poor generalization on this specific task.

## Files
- `model.py`: Bisine Network implementation with manual Gradient/Hessian.
- `optimizer.py`: SFNOptimizer with backtracking line search.
- `verify_math.py`: Verification of manual derivatives against PyTorch autograd.
- `compare.py`: Main experiment script.
- `results.png`: Training curves and eigenvalue plots.

## Conclusion
The Bisine Network provides a useful playground for second-order methods due to its structured Hessian. While SFN is powerful at finding training minima and handling saddle points, regularization (e.g., weight decay or stochasticity) is likely needed for better generalization on competitive benchmarks like `mnist1d`.
