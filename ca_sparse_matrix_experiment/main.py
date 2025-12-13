import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigs
import scipy.sparse

def run_cellular_automaton(rule, size, n_steps):
    """
    Runs a 1D elementary cellular automaton.
    - rule: The rule number (0-255).
    - size: The width of the automaton grid.
    - n_steps: The number of time steps to simulate.
    """
    rule_bin = format(rule, '08b')
    rule_map = {
        (1, 1, 1): int(rule_bin[0]), (1, 1, 0): int(rule_bin[1]),
        (1, 0, 1): int(rule_bin[2]), (1, 0, 0): int(rule_bin[3]),
        (0, 1, 1): int(rule_bin[4]), (0, 1, 0): int(rule_bin[5]),
        (0, 0, 1): int(rule_bin[6]), (0, 0, 0): int(rule_bin[7])
    }

    grid = np.zeros((n_steps, size), dtype=int)
    grid[0, size // 2] = 1

    for i in range(1, n_steps):
        for j in range(size):
            left = grid[i - 1, (j - 1 + size) % size]
            center = grid[i - 1, j]
            right = grid[i - 1, (j + 1) % size]
            grid[i, j] = rule_map.get((left, center, right), 0)
    return grid

def automaton_to_sparse_matrix(automaton_grid):
    """Converts the automaton grid to a sparse matrix."""
    return lil_matrix(automaton_grid)

def create_random_sparse_matrix(size, n_steps, density):
    """Creates a random sparse matrix with similar properties."""
    return scipy.sparse.random(n_steps, size, density=density, format='lil')

def get_eigenvalues(matrix):
    """Computes the eigenvalues of a matrix."""
    try:
        # k must be less than the matrix dimensions
        k = min(matrix.shape) - 2
        eigenvalues = eigs(matrix.asfptype(), k=k, which='LM', return_eigenvectors=False)
        return eigenvalues.real
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.array([])

def main():
    # Experiment parameters
    RULE = 30
    SIZE = 100
    N_STEPS = 100

    # Generate matrix from cellular automaton
    ca_grid = run_cellular_automaton(RULE, SIZE, N_STEPS)
    ca_matrix = automaton_to_sparse_matrix(ca_grid)
    density = ca_matrix.nnz / (SIZE * N_STEPS)

    # Generate random sparse matrix with the same density
    random_matrix = create_random_sparse_matrix(SIZE, N_STEPS, density)

    # Compute eigenvalues
    ca_eigenvalues = get_eigenvalues(ca_matrix)
    random_eigenvalues = get_eigenvalues(random_matrix)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.hist(ca_eigenvalues, bins=50, alpha=0.6, label=f'CA Rule {RULE}', color='blue')
    plt.hist(random_eigenvalues, bins=50, alpha=0.6, label='Random', color='red')
    plt.title('Eigenvalue Distribution')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.legend()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Save the plot in the same directory as the script
    plt.savefig(os.path.join(script_dir, 'eigenvalue_distribution.png'))
    plt.close()

if __name__ == '__main__':
    main()
