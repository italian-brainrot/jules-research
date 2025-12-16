
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse.linalg import cg, LinearOperator

def lanczos_sqrt_mv(A_op, b, n_steps):
    """
    Approximates A^{1/2}b using the Lanczos algorithm.
    """
    n = len(b)
    V = np.zeros((n, n_steps + 1))
    T = np.zeros((n_steps, n_steps))

    v = b / np.linalg.norm(b)
    V[:, 0] = v

    w_prime = A_op @ v
    alpha = np.dot(w_prime, v)
    w = w_prime - alpha * v
    T[0, 0] = alpha

    for j in range(1, n_steps):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            # Lucky breakdown, the subspace is invariant
            break

        v_prev = v
        v = w / beta
        V[:, j] = v

        w_prime = A_op @ v
        alpha = np.dot(w_prime, v)
        w = w_prime - alpha * v - beta * v_prev

        T[j, j] = alpha
        T[j, j-1] = beta
        T[j-1, j] = beta

    # T is now a tridiagonal matrix of size (j x j)
    # V is of size (n x j)
    # We need to truncate T and V if breakdown occurred
    T = T[:j, :j]
    V = V[:, :j]

    # Compute the square root of T
    eigvals, eigvecs = np.linalg.eigh(T)
    sqrt_eigvals = np.sqrt(eigvals)
    sqrt_T = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

    # Approximate A^{1/2}b
    # The first column of V is b/||b||, so V^T b = ||b|| * e_1
    # A^{1/2}b approx V @ sqrt(T) @ V.T @ b = V @ sqrt(T) @ ||b|| * e_1
    b_hat = np.linalg.norm(b) * V @ sqrt_T[:, 0]

    return b_hat

def preconditioned_richardson(A_op, b, P_inv, omega, tol=1e-6, max_iter=100):
    """
    Solves Ax=b using the Preconditioned Richardson method with a simple line search.
    """
    x = np.zeros_like(b)
    residuals = []

    for i in range(max_iter):
        r = b - (A_op @ x)
        residual_norm = np.linalg.norm(r)
        residuals.append(residual_norm)
        if residual_norm < tol:
            break

        z = P_inv(r)

        # Simple line search for omega
        current_omega = omega
        x_new = x + current_omega * z
        new_residual_norm = np.linalg.norm(b - (A_op @ x_new))

        # Backtrack if residual increases
        while new_residual_norm > residual_norm and current_omega > 1e-6:
            current_omega *= 0.5
            x_new = x + current_omega * z
            new_residual_norm = np.linalg.norm(b - (A_op @ x_new))

        x = x_new

    return x, residuals

def lanczos_cg(A_op, b, lanczos_steps, tol=1e-6, max_iter=100):
    b_hat = lanczos_sqrt_mv(A_op, b, n_steps=lanczos_steps)

    residuals = []
    def callback(xk):
        residuals.append(np.linalg.norm(b_hat - (A_op @ xk)))

    x, info = cg(A_op, b_hat, rtol=tol, maxiter=max_iter, callback=callback)
    return x, residuals

def prl_solver(A_op, b, P_inv, lanczos_steps, omega, tol=1e-6, max_iter=100):
    b_hat = lanczos_sqrt_mv(A_op, b, n_steps=lanczos_steps)
    x, residuals = preconditioned_richardson(A_op, b_hat, P_inv, omega, tol=tol, max_iter=max_iter)
    return x, residuals

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

    # Define the preconditioner inverse
    P_diag = np.diag(A)
    def P_inv(r):
        return r / P_diag

    # --- Parameters ---
    lanczos_steps = 20
    omega = 1.0 # Initial relaxation parameter for Richardson
    tol = 1e-8
    max_iter = 100

    # --- Run Solvers ---
    n = len(b)
    A_op = LinearOperator((n, n), matvec=A_mv)

    # Lanczos-CG (Baseline)
    x_cg, residuals_cg = lanczos_cg(A_op, b, lanczos_steps, tol=tol, max_iter=max_iter)

    # Preconditioned Richardson-Lanczos (New Method)
    x_prl, residuals_prl = prl_solver(A_op, b, P_inv, lanczos_steps, omega, tol=tol, max_iter=max_iter)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(np.arange(1, len(residuals_cg) + 1), residuals_cg, label='Lanczos-CG (Baseline)')
    plt.semilogy(np.arange(1, len(residuals_prl) + 1), residuals_prl, label=f'PRL ($\\omega$={omega})')
    plt.title('Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm ($||A\hat{x} - \hat{b}||_2$)')
    plt.legend()
    plt.grid(True)

    # Save the plot in the experiment's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'convergence_comparison.png'))
    plt.show()