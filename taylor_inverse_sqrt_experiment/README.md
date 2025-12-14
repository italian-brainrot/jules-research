# Experiment: Iterative Method for Inverse Square Root of a Symmetric Matrix

This experiment benchmarks the Newton-Raphson method, an iterative approach for computing the inverse square root of a symmetric real matrix, against standard decomposition-based methods.

## Methodology

Three methods were implemented and benchmarked:

1.  **Eigendecomposition-based:** This standard method computes the inverse square root by finding the eigenvalues and eigenvectors of the matrix.
2.  **SVD-based:** This method uses the Singular Value Decomposition of the matrix to compute the inverse square root.
3.  **Newton-Raphson Method:** An iterative method that refines an initial guess to converge to the inverse square root. The formula for the iteration is:
    `X_{k+1} = 0.5 * X_k * (3I - A * X_k^2)`

The benchmarking was performed on randomly generated symmetric positive-definite matrices of various sizes. The execution time and reconstruction error (measured by the Frobenius norm of the difference between the reconstructed inverse and the true inverse) were recorded for each method.

## Results

The benchmarking results are summarized in the plots below.

### Execution Time

![Execution Times](execution_times.png)

The plot shows that the Newton-Raphson method is competitive in terms of execution time, especially for a small number of iterations. As the matrix size increases, the eigendecomposition-based method becomes more efficient than the Newton-Raphson method with a higher number of iterations.

### Reconstruction Error

![Reconstruction Errors](reconstruction_errors.png)

The reconstruction error plot shows that the Newton-Raphson method is highly accurate and converges to a solution with an error close to machine precision, similar to the eigendecomposition and SVD methods. The accuracy improves with the number of iterations, as expected.

## Conclusion

The experiment demonstrates that the Newton-Raphson method is a viable and accurate iterative approach for computing the inverse square root of a symmetric matrix. It offers a good balance between performance and accuracy, especially when a moderate number of iterations are used.

For applications where high precision is required, the eigendecomposition-based method is generally the most efficient and reliable. However, the Newton-Raphson method provides a strong alternative, particularly in scenarios where matrix decompositions are undesirable or computationally expensive.
