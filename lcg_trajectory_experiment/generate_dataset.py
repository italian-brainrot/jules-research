
import torch
import matplotlib.pyplot as plt

def generate_spd_matrix(n):
    """Generates a symmetric positive-definite matrix of size n x n."""
    A = torch.randn(n, n)
    return torch.matmul(A, A.T) + torch.eye(n) * 1e-3

def conjugate_gradient(A, b, x0=None, max_iters=100, tol=1e-6):
    """
    Solves the linear system Ax = b using the Conjugate Gradient method.
    Records the history of the solution vector x at each iteration.
    """
    n = len(b)
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - torch.matmul(A, x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    x_history = [x.clone()]

    for i in range(max_iters):
        Ap = torch.matmul(A, p)
        alpha = rs_old / torch.dot(p, Ap)

        x += alpha * p
        r -= alpha * Ap

        rs_new = torch.dot(r, r)

        x_history.append(x.clone())

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, x_history

def plot_trajectories(x_history, num_coords_to_plot=5):
    """
    Plots the value of a few coordinates of the solution vector over iterations.
    """
    x_history_tensor = torch.stack(x_history)
    plt.figure(figsize=(12, 6))
    for i in range(min(num_coords_to_plot, x_history_tensor.shape[1])):
        plt.plot(x_history_tensor[:, i].numpy(), label=f'Coordinate {i}')

    plt.xlabel('Iteration')
    plt.ylabel('Coordinate Value')
    plt.title('Conjugate Gradient Solution Trajectories')
    plt.legend()
    plt.grid(True)
    plt.savefig('cg_trajectories.png')
    print("Saved trajectory plot to cg_trajectories.png")

if __name__ == '__main__':
    # --- Parameters ---
    matrix_size = 50
    max_iterations = 100

    # --- Generate Data ---
    A = generate_spd_matrix(matrix_size)
    b = torch.randn(matrix_size)

    # --- Run CG and get history ---
    x_solution, x_history = conjugate_gradient(A, b, max_iters=max_iterations)

    # --- Save history ---
    torch.save(x_history, 'lcg_trajectory_experiment/x_history.pt')
    print(f"Saved solution history of {len(x_history)} iterations to lcg_trajectory_experiment/x_history.pt")

    # --- Plot trajectories for visual inspection ---
    # We need to move the plot into the new directory
    plot_trajectories(x_history)
    import os
    os.rename('cg_trajectories.png', 'lcg_trajectory_experiment/cg_trajectories.png')
