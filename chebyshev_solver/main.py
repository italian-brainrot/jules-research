
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse.linalg import LinearOperator, cg

def lanczos_iteration(A_op, n_steps, v_start=None):
    """
    Performs the Lanczos iteration to generate a tridiagonal matrix T and an orthogonal basis Q.
    """
    n = A_op.shape[0]
    if v_start is None:
        q = np.random.rand(n)
        q /= np.linalg.norm(q)
    else:
        q = v_start / np.linalg.norm(v_start)

    Q = np.zeros((n, n_steps))
    alphas = []
    betas = []

    for j in range(n_steps):
        Q[:, j] = q
        v = A_op @ q
        alpha = np.dot(q, v)
        alphas.append(alpha)

        v = v - alpha * q
        if j > 0:
            v = v - betas[j-1] * Q[:, j-1]

        beta = np.linalg.norm(v)
        if beta > 1e-10 and j < n_steps - 1:
            betas.append(beta)
            q = v / beta
        else:
            break

    T = np.diag(alphas)
    if betas:
        T += np.diag(betas, k=1)
        T += np.diag(betas, k=-1)

    return T, Q[:, :len(alphas)]

def lanczos_sqrt_mv(A_op, b, n_steps):
    """
    Approximates A^{1/2}b using the Lanczos algorithm with eigendecomposition of T.
    """
    T, Q = lanczos_iteration(A_op, n_steps, v_start=b)

    eigvals, eigvecs = np.linalg.eigh(T)
    sqrt_T = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T

    e1 = np.zeros(T.shape[0])
    e1[0] = 1.0
    b_hat = np.linalg.norm(b) * Q @ (sqrt_T @ e1)

    return b_hat

def lanczos_cg_solver(A_op, b, lanczos_steps):
    """
    Solves Ax = A^{1/2}b using Lanczos approximation and CG.
    """
    b_hat = lanczos_sqrt_mv(A_op, b, n_steps=lanczos_steps)
    x, info = cg(A_op, b_hat)
    return x

def estimate_eigenvalues(A_op, n_steps=30):
    """
    Estimates the extremal eigenvalues of a linear operator A using the Lanczos algorithm.
    """
    T, _ = lanczos_iteration(A_op, n_steps)
    eigvals = np.linalg.eigvalsh(T)
    return np.min(eigvals), np.max(eigvals)

def chebyshev_sqrt_solver(A_op, b, lambda_min, lambda_max, degree=20):
    """
    Solves A^{-1/2}b using a Chebyshev polynomial approximation.
    THIS IMPLEMENTATION IS BUGGY AND DOES NOT WORK.
    """
    # Map the interval [lambda_min, lambda_max] to [-1, 1] for Chebyshev polynomials
    g = lambda y: (y * (lambda_max - lambda_min) + (lambda_max + lambda_min)) / 2
    f = lambda y: g(y)**(-0.5)

    # Compute coefficients of the Chebyshev expansion of f
    coeffs = np.polynomial.chebyshev.chebfit(np.linspace(-1, 1, 100), f(np.linspace(-1, 1, 100)), degree)

    # Clenshaw's algorithm for p(A)b
    w0 = b
    w1 = (A_op @ b - (lambda_max + lambda_min) / 2 * b) * (2 / (lambda_max - lambda_min))

    res = coeffs[0] * w0 + coeffs[1] * w1

    for i in range(2, degree + 1):
        wi = 2 * ((A_op @ w1 - (lambda_max + lambda_min) / 2 * w1) * (2 / (lambda_max - lambda_min))) - w0
        res = res + coeffs[i] * wi
        w0, w1 = w1, wi

    return res

def chebyshev_solver_main(A_op, b, lanczos_steps=30, chebyshev_degree=20):
    """
    Main solver that combines eigenvalue estimation and Chebyshev approximation.
    """
    lambda_min, lambda_max = estimate_eigenvalues(A_op, n_steps=lanczos_steps)

    # Add a small shift to avoid singularity
    lambda_min = max(lambda_min, 1e-6)

    x = chebyshev_sqrt_solver(A_op, b, lambda_min, lambda_max, degree=chebyshev_degree)
    return x

if __name__ == '__main__':
    # Create a synthetic SPD matrix
    n = 100
    np.random.seed(0)
    A = np.random.rand(n, n)
    A = A @ A.T + np.eye(n) * 0.1 # Ensure SPD

    # Create a random vector b
    b = np.random.rand(n)

    # Define the matrix-vector product function
    def A_mv(v):
        return A @ v

    A_op = LinearOperator((n, n), matvec=A_mv)

    # --- Parameters ---
    lanczos_steps = 30
    chebyshev_degree = 30

    # --- Run Solvers ---
    x_cheby = chebyshev_solver_main(A_op, b, lanczos_steps, chebyshev_degree)
    x_cg = lanczos_cg_solver(A_op, b, lanczos_steps)

    # --- Baseline (True Solution) ---
    from scipy.linalg import sqrtm
    A_sqrt_inv = np.linalg.inv(sqrtm(A))
    x_true = A_sqrt_inv @ b

    # --- Error Calculation ---
    error_cheby = np.linalg.norm(x_cheby - x_true) / np.linalg.norm(x_true)
    error_cg = np.linalg.norm(x_cg - x_true) / np.linalg.norm(x_true)
    print(f"Relative error of Chebyshev solver: {error_cheby:.6f}")
    print(f"Relative error of Lanczos-CG solver: {error_cg:.6f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_true, label='True Solution', linestyle='--')
    plt.plot(x_cheby, label=f'Chebyshev (err={error_cheby:.4f})', alpha=0.8)
    plt.plot(x_cg, label=f'Lanczos-CG (err={error_cg:.4f})', alpha=0.6)
    plt.title('Comparison of Solver Solutions')
    plt.xlabel('Vector Component Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'solution_comparison.png'))
    plt.show()
