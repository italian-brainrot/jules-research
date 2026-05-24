# Differentiable Variational Mode Decomposition (DVMD) Experiment

This experiment implements a differentiable version of Variational Mode Decomposition (VMD) as a neural network layer and evaluates its effectiveness for signal classification on the MNIST-1D dataset.

## Methodology

### Variational Mode Decomposition (VMD)
VMD is a signal decomposition technique that adaptively decomposes a signal into a number of band-limited Intrinsic Mode Functions (IMFs). Each mode is concentrated around a central frequency $\omega_k$. The decomposition is formulated as a constrained optimization problem:
$$\min_{\{u_k\}, \{\omega_k\}} \sum_k \| \partial_t [(\delta(t) + \frac{i}{\pi t}) * u_k(t)] e^{-i\omega_k t} \|_2^2 \text{ subject to } \sum_k u_k = f$$

### Differentiable Implementation
We implemented VMD using PyTorch, making the decomposition process differentiable. This allows:
1. Gradients to flow back to the input signal.
2. Learning of the balancing parameter $\alpha$ (which controls the bandwidth of the modes).
3. Learning of the number of modes (indirectly, or by tuning).

Our implementation uses the Alternating Direction Method of Multipliers (ADMM) in the spectral domain, iterated for a fixed number of steps (`n_iter`) to ensure differentiability.

### Features Extracted
From the decomposition, we extract two primary features for each mode:
1. **Mode Energy**: The $L^2$ norm of the mode in the spectral domain.
2. **Center Frequency**: The estimated central frequency $\omega_k$.

These features are then fed into an MLP for classification.

## Experimental Setup
- **Dataset**: MNIST-1D (5,000 samples for tuning/evaluation).
- **Models**:
    - `BaselineMLP`: A 2-layer MLP operating on raw signals.
    - `DVMDNet`: An MLP operating only on mode energies and center frequencies.
    - `DVMDAugmentedMLP`: An MLP operating on both raw signals and DVMD features.
- **Tuning**: Optuna was used to tune the learning rate and number of modes for each model.

## Results

| Model | Accuracy (%) |
|-------|--------------|
| BaselineMLP | 65.00% ± 0.59% |
| DVMDNet | 25.33% ± 0.39% |
| DVMDAugmentedMLP | 61.80% ± 0.80% |

## Conclusion
The differentiable VMD features (energies and frequencies) alone were not as discriminative as the raw signal for the MNIST-1D classification task, achieving only ~25% accuracy. Furthermore, augmenting the raw signal with these features did not provide a performance boost; in fact, it slightly degraded performance. This suggests that while VMD is powerful for signal analysis and denoising, the specific features extracted (energies and frequencies of a few modes) might not capture the nuances required for MNIST-1D as well as the raw temporal patterns do, or that the current differentiability through many ADMM iterations is noisy.
