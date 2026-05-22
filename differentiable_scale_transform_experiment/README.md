# Differentiable Scale Transform Experiment

This experiment explores the use of the **Differentiable Scale Transform** (Mellin Transform on an exponential scale) as a feature extractor for 1D signal classification on the MNIST-1D dataset.

## Motivation

The Scale Transform is a mathematical tool that is theoretically invariant to the scaling of a signal. Since MNIST-1D includes scaling as one of its transformations, a Scale Transform based representation should, in theory, provide scale-invariant features that could improve classification robustness and performance.

## Method

### Differentiable Scale Transform
We implemented a differentiable version of the Scale Transform by:
1.  **Exponential Resampling**: Interpolating the input signal $x(t)$ at points $t = e^u$ for uniformly spaced $u$.
2.  **Isometric Weighting**: Multiplying the resampled signal by $\sqrt{t}$ to preserve the $L^2$ norm.
3.  **Fourier Transform**: Computing the magnitude of the FFT of the log-sampled signal.

The implementation uses `torch.nn.functional.grid_sample` for differentiable linear interpolation.

### Fourier-Mellin Layer
We also implemented a simplified Fourier-Mellin transform which consists of:
1.  Computing the magnitude of the FFT (Shift Invariance).
2.  Applying the Scale Transform to the FFT magnitude (Scale Invariance).

### Models
We compared four models:
-   **BaselineMLP**: A 3-layer MLP acting on the raw signal.
-   **ScaleTransformMLP**: A 3-layer MLP acting only on the Scale Transform magnitude.
-   **ScaleTransformAugmentedMLP**: A 3-layer MLP acting on the concatenation of the raw signal and Scale Transform magnitude.
-   **FourierMellinMLP**: A 3-layer MLP acting on the Fourier-Mellin features.

All models were tuned using Optuna for their learning rate and evaluated over 3 seeds.

## Results

| Model | Test Accuracy (Mean +/- Std) | Best Learning Rate |
| :--- | :--- | :--- |
| **BaselineMLP** | 67.87% +/- 0.87% | 0.005686 |
| **ScaleTransformMLP** | 21.57% +/- 1.52% | 0.003286 |
| **ScaleTransformAugmentedMLP** | 43.60% +/- 1.34% | 0.003775 |
| **FourierMellinMLP** | 22.90% +/- 1.04% | 0.001694 |

## Analysis

1.  **Baseline Dominance**: The baseline MLP significantly outperformed all Scale Transform-based models.
2.  **Low Discriminative Power**: The Scale Transform magnitude (and Fourier-Mellin features) alone yielded very low accuracy (~22%). This suggests that while these features are theoretically scale-invariant, they discard too much discriminative information (e.g., phase information, absolute scale/position) that is crucial for distinguishing between digits in MNIST-1D.
3.  **Negative Impact of Augmentation**: Surprisingly, adding Scale Transform features to the raw signal (`ScaleTransformAugmentedMLP`) reduced performance compared to the baseline. This might be due to the increased input dimensionality making the optimization task harder or introducing noise that confuses the MLP.
4.  **Discretization Effects**: The Scale Transform's invariance is theoretical for continuous signals. In the discrete case with only 40 points (MNIST-1D), interpolation and finite windowing effects likely degrade the invariance and feature quality.

## Conclusion

While the Differentiable Scale Transform provides a mathematically elegant way to achieve scale invariance, its application to small, discrete signals like MNIST-1D did not yield performance improvements in this experimental setup. The loss of information associated with taking the magnitude of the transform appears to outweigh the benefits of scale invariance for this specific task.
