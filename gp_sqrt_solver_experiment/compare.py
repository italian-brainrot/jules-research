class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
import os

# Load the GP-evolved solver from main.py
def load_gp_solver():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    solver_path = os.path.join(script_dir, 'best_solver.pkl')
    with open(solver_path, 'rb') as f:
        return pickle.load(f)

# The evaluate function from main.py is needed to run the GP solver
def evaluate(node, A, b, x):
    if node.value == 'x':
        return x
    if node.value == 'b':
        return b
    if node.value == 'A*':
        return A @ evaluate(node.left, A, b, x)

    left_val = evaluate(node.left, A, b, x)
    right_val = evaluate(node.right, A, b, x)

    if node.value == '+':
        return left_val + right_val
    if node.value == '-':
        return left_val - right_val
    if node.value == '*':
        return left_val * right_val
    raise ValueError(f"Unknown operator: {node.value}")

def gp_solver_iteration(solver_node, A, b, x):
    return evaluate(solver_node, A, b, x)

# Baseline Solvers
def solve_eig(A, b):
    eigvals, eigvecs = np.linalg.eigh(A)
    sqrt_eigvals = np.sqrt(eigvals)
    A_sqrt_inv = eigvecs @ np.diag(1.0 / sqrt_eigvals) @ eigvecs.T
    return A_sqrt_inv @ b

def solve_cg(A, b):
    # Solve Ax = A^{1/2}b
    # First, compute A^{1/2}b
    eigvals, eigvecs = np.linalg.eigh(A)
    sqrt_eigvals = np.sqrt(eigvals)
    A_sqrt = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    b_transformed = A_sqrt @ b

    x, _ = cg(A, b_transformed)
    return x

def benchmark():
    matrix_sizes = [10, 20, 50, 100, 200]
    results = {
        'gp': {'times': [], 'residuals': []},
        'eig': {'times': [], 'residuals': []},
        'cg': {'times': [], 'residuals': []}
    }

    gp_solver_node = load_gp_solver()
    # The GP solver is an additive update
    print(f"Loaded GP Solver: x_k+1 = x_k + {gp_solver_node}")

    for n in matrix_sizes:
        # Generate a test problem
        A = np.random.rand(n, n)
        A = np.dot(A, A.T)
        b = np.random.rand(n)

        # --- Eigendecomposition Solver ---
        start_time = time.time()
        x_eig = solve_eig(A, b)
        eig_time = time.time() - start_time
        results['eig']['times'].append(eig_time)

        # Calculate residual for eig solver
        eigvals, eigvecs = np.linalg.eigh(A)
        A_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        residual_eig = np.linalg.norm(A_sqrt @ x_eig - b) / np.linalg.norm(b)
        results['eig']['residuals'].append(residual_eig)

        # --- Conjugate Gradient Solver ---
        start_time = time.time()
        x_cg = solve_cg(A, b)
        cg_time = time.time() - start_time
        results['cg']['times'].append(cg_time)
        residual_cg = np.linalg.norm(A_sqrt @ x_cg - b) / np.linalg.norm(b)
        results['cg']['residuals'].append(residual_cg)

        # --- GP Evolved Solver ---
        x_gp = np.zeros_like(b)
        start_time = time.time()
        for _ in range(25): # Same number of iterations as in fitness function
             update = gp_solver_iteration(gp_solver_node, A, b, x_gp)
             # Handle potential instability
             if np.isnan(update).any() or np.isinf(update).any():
                 break
             x_gp = x_gp + update
        gp_time = time.time() - start_time
        results['gp']['times'].append(gp_time)
        residual_gp = np.linalg.norm(A_sqrt @ x_gp - b) / np.linalg.norm(b)
        results['gp']['residuals'].append(residual_gp)

        print(f"Size {n}: Eig Time: {eig_time:.4f}, CG Time: {cg_time:.4f}, GP Time: {gp_time:.4f}")
        print(f"Size {n}: Eig Res: {residual_eig:.4e}, CG Res: {residual_cg:.4e}, GP Res: {residual_gp:.4e}")

    return results, matrix_sizes

def plot_results(results, matrix_sizes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(matrix_sizes, results['gp']['times'], 'o-', label='GP Solver')
    ax1.plot(matrix_sizes, results['eig']['times'], 's-', label='Eigendecomposition')
    ax1.plot(matrix_sizes, results['cg']['times'], '^-', label='Conjugate Gradient')
    ax1.set_xlabel('Matrix Size (n)')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Solver Execution Time')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(matrix_sizes, results['gp']['residuals'], 'o-', label='GP Solver')
    ax2.plot(matrix_sizes, results['eig']['residuals'], 's-', label='Eigendecomposition')
    ax2.plot(matrix_sizes, results['cg']['residuals'], '^-', label='Conjugate Gradient')
    ax2.set_xlabel('Matrix Size (n)')
    ax2.set_ylabel('Relative Residual Norm')
    ax2.set_title('Solver Accuracy')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'solver_comparison.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    results, matrix_sizes = benchmark()
    plot_results(results, matrix_sizes)
