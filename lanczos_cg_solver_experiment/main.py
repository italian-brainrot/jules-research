
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse.linalg import cg as scipy_cg, LinearOperator
import os

# --- Helper functions from previous experiment ---

def lanczos(A, v, m):
    """
    Performs m steps of the Lanczos iteration.
    Handles early termination correctly.
    """
    n = len(v)
    Q = np.zeros((n, m + 1))
    H = np.zeros((m, m))

    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return Q[:, :m], H

    Q[:, 0] = v / norm_v

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
    Handles zero vector input.
    """
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.zeros_like(v)

    Q, H = lanczos(A, v, m)
    if H.shape[0] == 0:
        return np.zeros_like(v)

    eigvals, eigvecs = eigh(H)
    H_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T
    return Q @ H_sqrt[:, 0] * norm_v

# --- New Implementation: Lanczos-CG Solver ---

def lanczos_cg_solver(A, b, x0, m, max_iter=100, tol=1e-6, true_x=None):
    """
    Solves A^{1/2}x = b using Conjugate Gradient where the matrix-vector
    product A^{1/2}v is approximated by a Lanczos projection.
    """
    x = x0.copy()

    mat_vec_op = lambda v: matrix_sqrt_mv(A, v, m)

    total_matvecs = 0
    errors = []
    matvec_counts = []

    r = b - mat_vec_op(x)
    total_matvecs += m
    p = r.copy()
    rs_old = np.dot(r, r)

    for i in range(max_iter):
        Ap = mat_vec_op(p)
        total_matvecs += m

        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < tol:
            if true_x is not None:
                error = np.linalg.norm(x - true_x)
                errors.append(error)
                matvec_counts.append(total_matvecs)
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

        if true_x is not None:
            error = np.linalg.norm(x - true_x)
            errors.append(error)
            matvec_counts.append(total_matvecs)

    return x, errors, matvec_counts

def main():
    # --- 1. Setup the Problem ---
    n = 256
    np.random.seed(0)
    _A = np.random.randn(n, n)
    A_mat = _A @ _A.T
    A = LinearOperator((n, n), matvec=lambda v: A_mat @ v, dtype=np.float64)

    x_true = np.random.randn(n)
    b = matrix_sqrt_mv(A, x_true, m=n)

    # --- 2. Baseline Solver: CG on the normal equations A x = A^{1/2} b ---
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
    scipy_cg(A, b_transformed, x0=x0, rtol=1e-6, maxiter=n, callback=cg_callback)

    # --- 3. Proposed Solver: Lanczos-CG ---
    plt.figure(figsize=(10, 7))
    plt.plot(cg_matvec_counts, cg_errors, label='Baseline: CG on Normal Equations', marker='o', linestyle='--')

    for m in [5, 10, 15]:
        x0 = np.zeros(n)
        _, errors, matvecs = lanczos_cg_solver(
            A, b, x0, m=m, max_iter=150, true_x=x_true, tol=1e-6
        )
        plt.plot(matvecs, errors, label=f'Lanczos-CG (m={m})', marker='x')

    # --- 4. Plotting ---
    plt.xlabel('Number of Matrix-Vector Products with A')
    plt.ylabel('Error Norm ||x_k - x_true||')
    plt.title('Convergence Comparison for Solving $A^{1/2}x = b$')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'convergence_comparison_lanczos_cg.png'))
    plt.close()

    print("Experiment finished. Plot saved to 'convergence_comparison_lanczos_cg.png'")

if __name__ == '__main__':
    main()
