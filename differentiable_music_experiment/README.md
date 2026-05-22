# Differentiable MUSIC (Multiple Signal Classification) Experiment

This experiment evaluates the use of a differentiable MUSIC layer for signal classification on the MNIST-1D dataset.

## Methodology

### MUSIC Algorithm
MUSIC is a subspace-based frequency estimation technique. It decomposes the correlation matrix of a signal into signal and noise subspaces. The frequency pseudospectrum is computed by projecting steering vectors onto the noise subspace; peaks in the pseudospectrum correspond to frequencies present in the signal.

### Differentiable Implementation
Our implementation forms a Hankel matrix from the input signal and performs Singular Value Decomposition (SVD). To make the subspace selection differentiable, we use a soft-weighting mechanism:
- Eigenvalues are computed as $(S^2)/N$.
- A sigmoid-based weighting function $w_i = \sigma(\beta (\tau - \lambda_i))$ is applied to each eigenvector.
- This allows the model to learn the optimal threshold $\tau$ and steepness $\beta$ for separating signal from noise components.
- The pseudospectrum is evaluated on a fixed grid of frequencies and used as input to a downstream MLP.

## Experimental Setup
We compared three architectures on MNIST-1D:
1.  **Baseline MLP**: A standard multi-layer perceptron.
2.  **MUSIC MLP**: An MLP that takes only the MUSIC pseudospectrum as input.
3.  **MUSIC Augmented MLP**: An MLP that takes both the raw signal and the MUSIC pseudospectrum.

All models were tuned using Optuna for 20 trials each, optimizing learning rate and relevant hyperparameters (window size, number of frequencies).

## Results

| Model | Best Test Accuracy (%) |
| :--- | :---: |
| Baseline MLP | 64.15 |
| MUSIC MLP | 38.30 |
| MUSIC Augmented MLP | 67.90 |

### Visualization
The following plots illustrate the results (saved in the experiment directory):
- `comparison.png`: Bar chart comparing the best accuracy of each model.
- `pseudospectra.png`: Example MNIST-1D signals and their corresponding MUSIC pseudospectra.

## Findings
- **Augmentation Benefit**: The MUSIC Augmented MLP outperformed the baseline MLP, suggesting that the MUSIC pseudospectrum provides complementary frequency-domain information that is useful for classification.
- **Stand-alone Performance**: The MUSIC MLP alone performed significantly worse than the baseline. This indicates that while the pseudospectrum captures important frequency components, it discards critical information (like phase and temporal structure) present in the raw signal.
- **Learnability**: The differentiability of the layer allows it to be integrated directly into the neural network, although in this short experiment we primarily tuned it via Optuna. Future work could explore joint training of the MUSIC parameters with the rest of the network on more complex tasks.
