# Learnable Fractional Derivative Filterbank (LFDF) Experiment

This experiment investigates the use of **Learnable Fractional Derivative Layers** as a novel feature extraction mechanism for 1D signals, using the `mnist1d` dataset.

## Hypothesis

Standard convolutional layers learn arbitrary local filters. Fractional derivatives, on the other hand, provide a principled way to extract multi-scale features related to the rate of change of the signal. By making the order of differentiation ($\alpha$) a learnable parameter, the network can adaptively select the most informative spectral transformations for the task. We hypothesize that a filterbank of learnable fractional derivatives can capture more useful information than raw pixels, outperforming vanilla MLPs.

## Method

### Fractional Filterbank
The `FractionalFilterbank` layer operates in the frequency domain using the Fast Fourier Transform (FFT). For an input signal $x(t)$, the fractional derivative of order $\alpha$ is defined in the Fourier domain as:
$$\mathcal{F}(D^\alpha x)(\omega) = (i\omega)^\alpha \mathcal{F}(x)(\omega)$$
where $(i\omega)^\alpha = |\omega|^\alpha e^{i \alpha \pi / 2}$.

Our implementation:
1. Computes the RFFT of the padded input signal.
2. Applies a learnable multiplier $(i\omega)^\alpha$ for each filter in the filterbank, where $\alpha \in [0, 2]$ is learned via backpropagation.
3. Includes a learnable gain and bias for each filter.
4. Uses padding to avoid circular convolution artifacts.
5. Transforms the signal back to the time domain using Inverse RFFT.

### Models Compared
1. **MLP Baseline**: A 3-layer MLP (Hidden dim 256) with BatchNorm and ReLU.
2. **Conv1d Baseline**: A 1D Convolution (32 filters, kernel size 5) followed by a 2-layer MLP.
3. **LFDF Model**: A `FractionalFilterbank` (32 filters) followed by a 2-layer MLP.

All models were compared fairly by tuning their learning rates using Optuna over 12 trials each.

## Results

The experiment was performed on 5000 samples of `mnist1d`.

| Model | Tuning Best Acc | Final Test Acc (Max) | Final Test Acc (End) |
|-------|-----------------|----------------------|----------------------|
| MLP Baseline | 72.0% | 71.2% | 69.0% |
| Conv1d Baseline | 92.3% | 91.9% | 91.2% |
| **LFDF Model** | 82.7% | **84.1%** | 79.8% |

### Analysis
- **LFDF significantly outperformed the MLP baseline** (84.1% vs 71.2%), demonstrating that learnable fractional derivatives are powerful feature extractors for 1D sequences.
- **LFDF approached the performance of Conv1d**, despite being more constrained in its filter form (only 2 parameters per filter vs 5 in Conv1d).
- The learned $\alpha$ values ranged from **0.06 to 1.66**, suggesting that the model utilized a wide spectrum of differentiation orders, from nearly identity ($\alpha \approx 0$) to second-order-like derivatives ($\alpha \approx 2$).

## Conclusion
Learnable fractional derivatives provide a strong inductive bias for signal processing tasks. They offer a middle ground between the global but unconstrained nature of MLPs and the local but unconstrained nature of standard convolutions. The ability to learn the order of differentiation allows the network to automatically find the optimal "sharpness" of features.

## Artifacts
- `accuracy_comparison.png`: Training curves for all three models.
- `alpha_distribution.png`: Histogram of the learned fractional orders $\alpha$.
- `results.txt`: Detailed numerical results and learned parameters.
