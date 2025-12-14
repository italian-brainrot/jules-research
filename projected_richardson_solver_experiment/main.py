
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse.linalg import cg, LinearOperator
import os

def lanczos(A, v, m):
    """
    Performs m steps of the Lanczos iteration.
    """
    n = len(v)
    Q = np.zeros((n, m + 1))
    H = np.zeros((m, m))

    Q[:, 0] = v / np.linalg.norm(v)

    beta = 0
    for j in range(m):
        w = A.matvec(Q[:, j])
        alpha = np.dot(w, Q[:, j])

        w = w - alpha * Q[:, j]
        if j > 0:
            w = w - beta * Q[:, j - 1]

        H[j, j] = alpha
        beta = np.linalg.norm(w)

        if beta < 1e-10:
            m_actual = j + 1
            return Q[:, :m_actual], H[:m_actual, :m_actual]

        Q[:, j + 1] = w / beta
        if j < m - 1:
            H[j + 1, j] = H[j, j + 1] = beta

    return Q[:, :m], H

def matrix_sqrt_mv(A, v, m):
    """
    Approximates A^{1/2}v using a Lanczos-based projection.
    """
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.zeros_like(v)

    Q, H = lanczos(A, v, m)
    eigvals, eigvecs = eigh(H)
    H_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T
    return Q @ H_sqrt[:, 0] * norm_v

def projected_richardson_solver(A, b, x0, m, alpha, max_iter=100, tol=1e-6, true_x=None):
    """
    Solves A^{1/2}x = b using a Projected Richardson Iteration.
    """
    x = x0.copy()
    errors = []
    matvec_counts = []
    total_matvecs = 0

    for i in range(max_iter):
        residual = b - matrix_sqrt_mv(A, x, m)
        total_matvecs += m
        x = x + alpha * residual

        if true_x is not None:
            error = np.linalg.norm(x - true_x)
            errors.append(error)
            matvec_counts.append(total_matvecs)
            if error < tol:
                break

    return x, errors, matvec_counts

def main():
    # --- 1. Setup the Problem ---
    n = 256
    np.random.seed(0)
    # Generate a random symmetric positive definite matrix A
    _A = np.random.randn(n, n)
    A_mat = _A @ _A.T
    A = LinearOperator((n, n), matvec=lambda v: A_mat @ v, dtype=np.float64)

    # Generate the true solution x_true
    x_true = np.random.randn(n)

    # Compute b = A^{1/2}x_true with high precision
    # For a fair benchmark, we use a large Krylov subspace to get a "ground truth" b
    b = matrix_sqrt_mv(A, x_true, m=n)

    # --- 2. Baseline Solver: CG on the normal equations A x = A^{1/2} b ---
    # We need A^{1/2}b for the RHS. We compute this once with high precision.
    b_transformed = matrix_sqrt_mv(A, b, m=n)

    cg_errors = []
    cg_matvec_counts = []
    cg_matvec_counter = 0
    def cg_callback(xk):
        nonlocal cg_matvec_counter
        cg_matvec_counter += 1
        error = np.linalg.norm(xk - x_true)
        cg_errors.append(error)
        cg_matvec_counts.append(cg_matvec_counter)

    x0 = np.zeros(n)
    cg_sol, _ = cg(A, b_transformed, x0=x0, rtol=1e-6, maxiter=n, callback=cg_callback)

    # --- 3. Proposed Solver: Projected Richardson ---
    plt.figure(figsize=(10, 7))
    plt.plot(cg_matvec_counts, cg_errors, label='CG on Normal Equations', marker='o', linestyle='--')

    # Run for different subspace dimensions
    for m in [5, 10, 15]:
        x0 = np.zeros(n)
        pr_sol, pr_errors, pr_matvecs = projected_richardson_solver(
            A, b, x0, m=m, alpha=0.1, max_iter=150, true_x=x_true
        )
        plt.plot(pr_matvecs, pr_errors, label=f'Projected Richardson (m={m})', marker='x')

    # --- 4. Plotting ---
    plt.xlabel('Number of Matrix-Vector Products with A')
    plt.ylabel('Error Norm ||x_k - x_true||')
    plt.title('Convergence Comparison for Solving $A^{1/2}x = b$')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'convergence_comparison.png'))
    plt.close()

    print("Experiment finished. Plot saved to 'convergence_comparison.png'")

if __name__ == '__main__':
    main()
