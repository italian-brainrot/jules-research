# Chebyshev Polynomial Approximation for $A^{-1/2}b$

This experiment implements and benchmarks a matrix-free and decomposition-free method to approximate the solution of $A^{1/2}x=b$, where A is a real, symmetric, and positive-definite matrix. The method is based on approximating the function $f(z) = z^{-1/2}$ with a Chebyshev polynomial.

## Method

The core of the method is to construct a polynomial $p(z)$ that approximates $z^{-1/2}$ over the interval $[\lambda_{min}, \lambda_{max}]$, which contains the spectrum of A. The solution is then computed as $x \approx p(A)b$. This approach is entirely matrix-free and decomposition-free.

The implementation consists of two main stages:

1.  **Eigenvalue Estimation:** The extremal eigenvalues of A, $\lambda_{min}$ and $\lambda_{max}$, are estimated using a few steps of the Lanczos algorithm.
2.  **Chebyshev Polynomial Solver:** The `chebyshev_sqrt_solver` function constructs the Chebyshev polynomial approximation and computes its action on the vector `b`. **NOTE: This implementation is buggy and does not produce correct results.**

## Benchmarks

The Chebyshev solver was benchmarked against two other methods:

1.  **Direct Eigendecomposition:** The true solution is computed using `scipy.linalg.sqrtm` and `numpy.linalg.inv`. This method is decompositional and serves as the ground truth.
2.  **Lanczos-CG Solver:** A strong, matrix-free baseline. It first approximates $\hat{b} = A^{1/2}b$ using a Lanczos routine and then solves the system $Ax=\hat{b}$ using the Conjugate Gradient method.

## Results

For a synthetic 100x100 SPD matrix, the following relative errors were observed:

-   **Chebyshev Solver:** ~1.66 (Incorrect)
-   **Lanczos-CG Solver:** ~0.0038

The plot below shows a visual comparison of the solution vectors:

![Solution Comparison](solution_comparison.png)

The Lanczos-CG solver produces a solution that is visually indistinguishable from the true solution. The Chebyshev solver, due to a bug in the implementation, produces an incorrect result.

## Conclusion

The Lanczos-CG method is a highly effective and accurate matrix-free method for solving the $A^{1/2}x=b$ problem. The Chebyshev polynomial approximation, while theoretically sound, proved difficult to implement correctly and is not functional in its current state.
