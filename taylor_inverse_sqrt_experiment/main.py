import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import svd
import os

def inv_sqrt_eig(A):
    """Computes the inverse square root of a symmetric matrix using eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T

def inv_sqrt_svd(A):
    """Computes the inverse square root of a symmetric matrix using SVD."""
    U, s, Vh = svd(A)
    return U @ np.diag(1.0 / np.sqrt(s)) @ Vh

def inv_sqrt_newton(A, num_iter=10):
    """
    Computes the inverse square root of a symmetric matrix using the Denman-Beavers iteration.
    """
    Y = A
    Z = np.eye(A.shape[0])

    for _ in range(num_iter):
        Y_inv = np.linalg.inv(Y)
        Z_inv = np.linalg.inv(Z)
        Y = 0.5 * (Y + Z_inv)
        Z = 0.5 * (Z + Y_inv)

    return Z

def generate_spd_matrix(n):
    """Generates a symmetric positive-definite matrix of size n x n."""
    A = np.random.rand(n, n)
    return A @ A.T + np.eye(n) * 1e-3

def benchmark():
    """Benchmarks the implemented algorithms."""
    matrix_sizes = [10, 20, 50, 100, 200, 500]
    num_iter_list = [5, 10, 20]

    results = {
        'eig': {'times': [], 'errors': []},
        'svd': {'times': [], 'errors': []},
    }
    for num_iter in num_iter_list:
        results[f'newton_{num_iter}'] = {'times': [], 'errors': []}

    for n in matrix_sizes:
        print(f"Benchmarking for matrix size: {n}x{n}")
        A = generate_spd_matrix(n)
        A_inv = np.linalg.inv(A)

        # Benchmark eig
        start_time = time.time()
        inv_sqrt_A_eig = inv_sqrt_eig(A)
        end_time = time.time()
        results['eig']['times'].append(end_time - start_time)
        results['eig']['errors'].append(np.linalg.norm(inv_sqrt_A_eig @ inv_sqrt_A_eig - A_inv))

        # Benchmark svd
        start_time = time.time()
        inv_sqrt_A_svd = inv_sqrt_svd(A)
        end_time = time.time()
        results['svd']['times'].append(end_time - start_time)
        results['svd']['errors'].append(np.linalg.norm(inv_sqrt_A_svd @ inv_sqrt_A_svd - A_inv))

        # Benchmark newton
        for num_iter in num_iter_list:
            start_time = time.time()
            inv_sqrt_A_newton = inv_sqrt_newton(A, num_iter=num_iter)
            end_time = time.time()
            results[f'newton_{num_iter}']['times'].append(end_time - start_time)
            results[f'newton_{num_iter}']['errors'].append(np.linalg.norm(inv_sqrt_A_newton @ inv_sqrt_A_newton - A_inv))

    return matrix_sizes, results

def plot_results(matrix_sizes, results):
    """Plots the benchmarking results."""

    script_dir = os.path.dirname(os.path.abspath(__file__))

    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(matrix_sizes, data['times'], marker='o', label=method)
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time of Matrix Inverse Square Root Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, "execution_times.png"))

    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(matrix_sizes, data['errors'], marker='o', label=method)
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Reconstruction Error (Frobenius Norm)")
    plt.title("Reconstruction Error of Matrix Inverse Square Root Methods")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, "reconstruction_errors.png"))

if __name__ == '__main__':
    matrix_sizes, results = benchmark()
    plot_results(matrix_sizes, results)
