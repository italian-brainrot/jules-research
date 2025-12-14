
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import polar, inv

def generate_matrix_stream(n_steps=100, n_dim=10, step_size=0.01, noise_level=0.01):
    """
    Generates a stream of slowly changing random matrices with some noise.
    """
    # Start with a well-conditioned matrix
    A = np.random.randn(n_dim, n_dim)
    U, _, V = np.linalg.svd(A)
    s = np.linspace(1, 0.5, n_dim)
    A = U @ np.diag(s) @ V.T

    stream = [A]
    for _ in range(n_steps - 1):
        # Add a small random perturbation to drift the matrix
        drift = step_size * np.random.randn(n_dim, n_dim)
        A = A + drift
        # Add some observation noise
        noise = noise_level * np.random.randn(n_dim, n_dim)
        stream.append(A + noise)
    return stream

def main():
    # Experiment parameters
    N_STEPS = 200
    N_DIM = 10
    STEP_SIZE = 0.005  # Reduced to prevent divergence
    NOISE_LEVEL = 0.01 # Reduced to prevent divergence
    EMA_ALPHA = 0.3

    # Generate matrix stream
    A_stream = generate_matrix_stream(n_steps=N_STEPS, n_dim=N_DIM, step_size=STEP_SIZE, noise_level=NOISE_LEVEL)

    # --- WSOI Tracker ---
    # The state we track is Y_k ≈ (A_k^T A_k)^(-1/2)
    # Initialize Y_0 from the true polar decomposition of the first matrix
    _, P_init = polar(A_stream[0])
    Y = inv(P_init)

    wsoi_errors = []
    for A in A_stream:
        M = A.T @ A
        # One iteration of Newton-Schulz for M^(-1/2), warm-started with Y from the previous step.
        # This iteration converges quadratically if the initial guess is good.
        Y = 0.5 * Y @ (3 * np.eye(M.shape[0]) - M @ Y @ Y)

        # The matrix sign U is recovered from A and Y
        U = A @ Y

        U_true, _ = polar(A)
        wsoi_errors.append(np.linalg.norm(U - U_true, 'fro'))

    # --- Baseline Tracker (EMA) ---
    A_ema = A_stream[0]
    baseline_errors = []
    for A in A_stream:
        # Apply exponential moving average to the matrices themselves
        A_ema = EMA_ALPHA * A + (1 - EMA_ALPHA) * A_ema

        # Compute the polar decomposition of the smoothed matrix
        U_ema, _ = polar(A_ema)

        U_true, _ = polar(A)
        baseline_errors.append(np.linalg.norm(U_ema - U_true, 'fro'))

    # Plot results
    plt.figure(figsize=(12, 7))
    plt.plot(wsoi_errors, label='WSOI Tracker (proposed)', linewidth=2)
    plt.plot(baseline_errors, label=f'Baseline Tracker (EMA, alpha={EMA_ALPHA})', linestyle='--', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Frobenius Norm Error')
    plt.title('WSOI vs. EMA Baseline for Tracking Matrix Sign')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.yscale('log')
    plt.tight_layout()

    # Save the plot
    output_path = 'matrix_sign_experiment/tracking_comparison.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
