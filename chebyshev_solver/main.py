
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg

def lanczos_sqrt_mv(A_op, b, n_steps):
    """
    Approximates A^{1/2}b using the Lanczos algorithm with eigendecomposition of T.
    """
    n = len(b)
    V = np.zeros((n, n_steps))
    T = np.zeros((n_steps, n_steps))

    v = b / np.linalg.norm(b)
    V[:, 0] = v

    w_prime = A_op @ v
    alpha = np.dot(w_prime, v)
    w = w_prime - alpha * v
    T[0, 0] = alpha

    dim = n_steps
    for j in range(1, n_steps):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            dim = j # Lucky breakdown
            break

        v_prev = v
        v = w / beta
        V[:, j] = v

        w_prime = A_op @ v
        alpha = np.dot(w_prime, v)
        w = w_prime - alpha * v - beta * v_prev

        T[j, j] = alpha
        if j < n_steps:
            T[j, j-1] = beta
            T[j-1, j] = beta

    T = T[:dim, :dim]
    V = V[:, :dim]

    eigvals, eigvecs = np.linalg.eigh(T)
    sqrt_T = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    e1 = np.zeros(dim)
    e1[0] = 1.0
    b_hat = np.linalg.norm(b) * V @ (sqrt_T @ e1)

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
    This is a decomposition-free method.
    """
    n = A_op.shape[0]
    q = np.random.rand(n)
    q /= np.linalg.norm(q)

    alpha, beta = 0, 0
    T = np.zeros((n_steps, n_steps))

    for j in range(n_steps):
        v = q
        q = A_op @ q
        if j > 0:
            q -= beta * v_prev
        alpha = np.dot(q, v)
        q -= alpha * v
        beta = np.linalg.norm(q)

        T[j, j] = alpha
        if j < n_steps - 1:
            T[j, j+1] = beta
            T[j+1, j] = beta

        if beta < 1e-10:
            break

        v_prev = v
        q /= beta

    # Eigenvalues of T are approximations of eigenvalues of A
    eigvals = np.linalg.eigvalsh(T[:j+1, :j+1])
    return np.min(eigvals), np.max(eigvals)

def chebyshev_sqrt_solver(A_op, b, lambda_min, lambda_max, degree=20):
    """
    Solves A^{-1/2}b using a Chebyshev polynomial approximation.
    This is a decomposition-free and matrix-free method.
    """
    # Map the interval [lambda_min, lambda_max] to [-1, 1]
    def map_x(x):
        return (2 * x - (lambda_max + lambda_min)) / (lambda_max - lambda_min)

    # Sample points for Chebyshev interpolation
    nodes = np.cos(np.pi * (np.arange(degree) + 0.5) / degree)

    # Inverse map from [-1, 1] to [lambda_min, lambda_max]
    def inv_map_x(y):
        return 0.5 * ((lambda_max - lambda_min) * y + (lambda_max + lambda_min))

    # Evaluate f(x) = x^{-1/2} on the mapped nodes
    f_vals = inv_map_x(nodes)**(-0.5)

    # Compute Chebyshev coefficients for f(x)
    coeffs = np.fft.fft(f_vals) / degree
    coeffs = np.real(coeffs[:degree])
    coeffs[0] /= 2.0

    # Evaluate the polynomial p(A)b using Clenshaw's algorithm
    y0 = b
    y1 = (A_op @ y0 - 0.5 * (lambda_max + lambda_min) * y0) * (2 / (lambda_max - lambda_min))

    x = coeffs[0] * y0
    for i in range(1, degree):
        x += coeffs[i] * y1
        y2 = 2 * (A_op @ y1 - 0.5 * (lambda_max + lambda_min) * y1) * (2 / (lambda_max - lambda_min)) - y0
        y0, y1 = y1, y2

    return x

def chebyshev_solver_main(A_op, b, lanczos_steps=30, chebyshev_degree=20):
    """
    Main solver that combines eigenvalue estimation and Chebyshev approximation.
    """
    # 1. Estimate extremal eigenvalues
    lambda_min, lambda_max = estimate_eigenvalues(A_op, n_steps=lanczos_steps)

    # 2. Solve using Chebyshev polynomial
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
    chebyshev_degree = 50

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
