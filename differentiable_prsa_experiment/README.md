# Differentiable Phase-Rectified Signal Averaging (DPRSA)

This experiment introduces a differentiable implementation of **Phase-Rectified Signal Averaging (PRSA)**, a technique traditionally used in medical signal processing (e.g., heart rate variability analysis) to extract periodic components from noisy, non-stationary signals.

## Methodology

The core idea of PRSA is to identify "anchor points" based on a specific criterion (e.g., an increase in value) and average segments of the signal centered at these points. This process synchronizes the phase of the components of interest while averaging out asynchronous noise and artifacts.

### Differentiable PRSA Layer (`DPRSALayer`)

Our differentiable implementation (`DPRSALayer`) makes the following components learnable:
1.  **Anchor Identification**: Instead of a fixed criterion, we use a 1D convolutional filter to compute an "anchorness" score at each time point.
2.  **Soft Gating**: We use a sigmoid (or softmax) function on the anchorness scores to create a differentiable weighting for each time point.
3.  **Weighted Averaging**: The output is computed as a weighted average of all segments (windows) of the signal, where weights are the anchorness scores.

Given a signal $x$ of length $L$, and learned anchor scores $s$, the output $O$ (a "rectified" segment of length $W$) is:
$$O = \frac{\sum_{t=1}^{L} s_t \cdot x_{t-\frac{W}{2} : t+\frac{W}{2}}}{\sum_{t=1}^{L} s_t}$$

## Experiments

We evaluated the DPRSA layer on the **MNIST-1D** dataset, which consists of 1D signals representing digits with various transformations, including translations.

### Models Compared
1.  **BaselineMLP**: A 2-layer MLP processing the raw 1D signal.
2.  **DPRSANet**: A model that first passes the signal through a `DPRSALayer` (8 anchors, window size 20) and then processes the resulting rectified segment with an MLP.
3.  **DPRSAAugmentedMLP**: An MLP that takes the concatenation of the raw signal and the DPRSA rectified segment.

### Training Details
-   Tuned learning rate for each model using Optuna (10 trials each).
-   Trained for 15 epochs per trial.
-   Dataset size: 10,000 samples.

## Results

| Model | Test Accuracy (%) |
| :--- | :--- |
| **BaselineMLP** | 78.30% |
| **DPRSANet** | **96.85%** |
| **DPRSAAugmentedMLP** | 93.90% |

DPRSA significantly outperformed the baseline, achieving nearly **97% accuracy**. This demonstrates that the DPRSA layer effectively learns to synchronize and extract discriminative patterns from the signal, providing a much cleaner and more stable representation than the raw, unaligned signal.

## Translation Invariance

Our logic tests confirmed that the DPRSA output is significantly more stable under signal translations compared to the raw signal. By learning to "anchor" the segments, the layer provides a form of learned translation invariance.

## Conclusion

Differentiable Phase-Rectified Signal Averaging is a powerful addition to neural network architectures for 1D signal processing. It successfully bridges classical signal processing insights with modern deep learning, enabling the model to learn optimal alignment and feature extraction strategies end-to-end.
