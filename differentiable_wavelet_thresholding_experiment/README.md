# Differentiable Adaptive Wavelet Thresholding (DAWT) Experiment

This experiment investigates the effectiveness of a **Differentiable Adaptive Wavelet Thresholding (DAWT)** layer as a preprocessing step for 1D signal classification on the `mnist1d` dataset.

## Hypothesis
Wavelet thresholding is a powerful classical technique for signal denoising. By integrating it into a neural network as a differentiable layer with learnable thresholds, we can provide a strong inductive bias for noise reduction that is specifically tuned to the classification task. We hypothesize that this layer will improve robustness to noise compared to a standard MLP.

## Method
The `DAWTLayer` implements a multi-level Discrete Wavelet Transform (DWT) using Haar wavelets.
- **Decomposition**: The signal is decomposed into approximation and detail coefficients using fixed Haar filters via 1D convolutions.
- **Thresholding**: Learnable soft-thresholding $\text{sgn}(x) \cdot \max(0, |x| - \tau)$ is applied to the detail coefficients at each level, where $\tau$ is a learnable parameter.
- **Reconstruction**: The denoised signal is reconstructed using the Inverse Discrete Wavelet Transform (IDWT) via 1D transposed convolutions.
- **Architecture**: The `DAWT-MLP` consists of a `DAWTLayer` followed by a standard MLP backbone. The `Baseline MLP` uses the same backbone without the `DAWTLayer`.

## Experimental Setup
- **Dataset**: MNIST-1D (10,000 samples).
- **Evaluation**: 50 epochs, 3 random seeds.
- **Noise Robustness**: Models were also evaluated on a noisy version of the test set (Gaussian noise with $\sigma=0.3$).
- **Hyperparameter Tuning**: Optuna was used to tune the learning rate, weight decay, and the number of wavelet levels (1-3) for each model (10 trials per model).

## Results

| Model | Clean Test Accuracy | Noisy Test Accuracy ($\sigma=0.3$) |
|-------|---------------------|-----------------------------------|
| Baseline MLP | **77.97% +/- 0.19%** | **70.12% +/- 0.66%** |
| DAWT-MLP | 76.78% +/- 0.78% | 69.00% +/- 0.92% |

### Learned Parameters (DAWT)
- **Best Levels**: 1
- **Mean Learned Threshold**: ~0.185 (level 1)

## Findings
1. **Competitive Performance**: The DAWT-augmented MLP achieved performance close to the baseline MLP, suggesting that the wavelet-based denoising inductive bias is compatible with the classification task.
2. **Learned Denoising**: The model learned a non-zero threshold for the detail coefficients (~0.185), demonstrating that the network automatically discovered a level of denoising that it found beneficial for classification.
3. **No Direct Robustness Gain**: In this specific configuration on `mnist1d`, the DAWT layer did not outperform the baseline MLP in either clean or noisy accuracy. This might be because the baseline MLP is already somewhat robust or the Haar wavelet is too simple for the complex patterns in `mnist1d`.
4. **Differentiability**: The experiment successfully demonstrated that classical wavelet-based denoising can be made fully differentiable and integrated into end-to-end learning.

## Visualizations
The `comparison.png` plot shows the accuracy comparison and a sample of the denoising effect learned by the DAWT layer.
