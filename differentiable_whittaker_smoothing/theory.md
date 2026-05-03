# Whittaker-Eilers Smoothing Theory

Whittaker-Eilers smoothing (also known as the "perfect smoother") is a penalized least squares method for signal smoothing. It balances the fidelity to the original data with the smoothness of the result.

## Objective Function

Given a noisy signal $y$ of length $N$, we want to find a smooth signal $z$ that minimizes:

$$Q = \sum_{i=1}^N (y_i - z_i)^2 + \lambda \sum_{i=d+1}^N (\Delta^d z_i)^2$$

In matrix form:

$$Q = \|y - z\|^2 + \lambda \|D_d z\|^2$$

where:
- $y$ is the noisy input signal.
- $z$ is the smoothed output signal.
- $\lambda$ is the smoothing parameter (higher $\lambda$ means more smoothing).
- $D_d$ is the $d$-th order difference matrix.

## Solution

To find the minimum, we take the derivative with respect to $z$ and set it to zero:

$$\frac{\partial Q}{\partial z} = -2(y - z) + 2\lambda D_d^T D_d z = 0$$
$$(I + \lambda D_d^T D_d) z = y$$

The solution is a linear system:

$$z = (I + \lambda D_d^T D_d)^{-1} y$$

Let $A = I + \lambda D_d^T D_d$. Then $A z = y$.

## Differentiable Formulation

In a neural network context, we want to be able to backpropagate through this smoothing operation to learn $\lambda$ or to use it as a pre-processing layer.

Since $A$ is a function of $\lambda$ and $z$ is a function of $A$ and $y$, we can backpropagate through the linear solver.
In PyTorch, `torch.linalg.solve(A, y)` is differentiable with respect to both $A$ and $y$.

### Difference Matrices

- $d=1$: $\Delta z_i = z_i - z_{i-1}$
- $d=2$: $\Delta^2 z_i = z_i - 2z_{i-1} + z_{i-2}$ (This is the most common choice, equivalent to Hodrick-Prescott filtering).
- $d=3$: $\Delta^3 z_i = z_i - 3z_{i-1} + 3z_{i-2} - z_{i-3}$

The matrix $D_d$ has dimensions $(N-d) \times N$.
The matrix $D_d^T D_d$ is $N \times N$ and is sparse (banded).

## Implementation Details

For small $N$ (like $N=40$ in `mnist1d`), we can use dense matrix operations. For very large $N$, sparse solvers would be more efficient, but `torch.linalg.solve` on dense matrices is sufficient for our experiments and benefits from GPU acceleration.
