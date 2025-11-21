
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def generate_spd_matrix(n):
    """Generates a symmetric positive-definite matrix of size n x n."""
    A = torch.randn(n, n)
    return torch.matmul(A, A.T) + torch.eye(n) * 1e-3

def conjugate_gradient(A, b, x0=None, max_iters=100, tol=1e-6):
    """Standard Conjugate Gradient solver."""
    n = len(b)
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    r = b - torch.matmul(A, x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    residual_history = [torch.norm(r)]

    for i in range(max_iters):
        Ap = torch.matmul(A, p)
        alpha = rs_old / torch.dot(p, Ap)

        x += alpha * p
        r -= alpha * Ap

        rs_new = torch.dot(r, r)
        residual_history.append(torch.sqrt(rs_new))

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, residual_history

def exponential_decay_model(t, a, b, c):
    """Model for the trajectory of a single coordinate."""
    return a * np.exp(-b * t) + c

def accelerated_cg(A, b, x0=None, max_iters=100, tol=1e-6, warmup_iters=15):
    """Accelerated CG with curve fitting and extrapolation."""
    n = len(b)
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    r = b - torch.matmul(A, x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    x_history = [x.clone()]
    residual_history = [torch.norm(r)]

    # 1. Warm-up phase
    for i in range(warmup_iters):
        Ap = torch.matmul(A, p)
        alpha = rs_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)

        x_history.append(x.clone())
        residual_history.append(torch.sqrt(rs_new))

        if torch.sqrt(rs_new) < tol:
            return x, residual_history

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    # 2. Extrapolation phase
    x_history_tensor = torch.stack(x_history)
    predicted_x = torch.zeros_like(x)

    for j in range(n):
        y_data = x_history_tensor[:, j].numpy()
        t_data = np.arange(len(y_data))

        try:
            # Fit the model and predict the final value (c)
            params, _ = curve_fit(exponential_decay_model, t_data, y_data, p0=[y_data[0], 0.1, y_data[-1]], maxfev=10000)
            predicted_x[j] = params[2] # c is the predicted final value
        except RuntimeError:
            # If fitting fails, just use the last known value as prediction
            predicted_x[j] = y_data[-1]

    # 3. Jump and restart CG
    x = predicted_x
    r = b - torch.matmul(A, x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    residual_history.append(torch.norm(r))

    # 4. Continue CG from the new starting point
    for i in range(warmup_iters, max_iters):
        Ap = torch.matmul(A, p)
        alpha = rs_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)

        residual_history.append(torch.sqrt(rs_new))

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, residual_history

def plot_loss_comparison(standard_residuals, accelerated_residuals):
    """Plots the residual norm vs. iteration for both algorithms."""
    plt.figure(figsize=(12, 6))
    plt.plot(standard_residuals, label='Standard CG')
    plt.plot(accelerated_residuals, label=f'Accelerated CG (Warmup: {len(accelerated_residuals) - len(standard_residuals) + 15})')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm ||b - Ax||')
    plt.title('Standard CG vs. Accelerated CG Convergence')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('lcg_trajectory_experiment/loss_comparison.png')
    print("Saved loss comparison plot to lcg_trajectory_experiment/loss_comparison.png")

if __name__ == '__main__':
    # --- Parameters ---
    matrix_size = 50
    max_iterations = 100
    warmup_iterations = 15

    # --- Generate Data ---
    A = generate_spd_matrix(matrix_size)
    b = torch.randn(matrix_size)

    # --- Run and Compare ---
    _, standard_cg_residuals = conjugate_gradient(A, b, max_iters=max_iterations)
    _, accelerated_cg_residuals = accelerated_cg(A, b, max_iters=max_iterations, warmup_iters=warmup_iterations)

    # --- Plot Results ---
    plot_loss_comparison(standard_cg_residuals, accelerated_cg_residuals)
