# Spectral Gated Whitening (SGW) Experiment

This experiment introduces **Spectral Gated Whitening (SGW)**, a novel differentiable preprocessing layer that learns to adaptively filter the spectral components of data during the whitening process.

## Hypothesis
Standard ZCA whitening ($W = U \Lambda^{-1/2} U^T$) decorrelates data but can severely amplify noise by boosting components with small eigenvalues. While techniques like Soft-PCA or shrinkage exist, they often rely on fixed functional forms and manual tuning of regularization parameters. SGW learns a task-specific spectral gate $s(\lambda) \in [0, 1]$ parameterized by a small MLP, allowing it to adaptively suppress noise-dominated components while maintaining the benefits of decorrelation for signal-rich components.

## Method
SGW computes the Eigenvalue Decomposition (EVD) of the training covariance matrix and applies the following transformation:
$$x' = U \text{diag}\left(\frac{s(\lambda_i)}{\sqrt{\lambda_i + \epsilon}}\right) U^T (x - \mu)$$
where $s(\lambda) = \sigma(\text{MLP}(\log \lambda))$. The MLP is trained end-to-end with the classifier.

## Experimental Setup
- **Dataset**: MNIST-1D
- **Task**: 10-class classification
- **Preprocessing Modes**:
  - `none`: No whitening.
  - `zca`: Standard ZCA whitening.
  - `soft_pca`: Fixed Wiener-like gate $s(\lambda) = \frac{\lambda}{\lambda + \eta}$ (tuned $\eta$).
  - `sgw`: Learnable Spectral Gated Whitening.
- **Tuning**: Optuna was used to tune learning rate, weight decay, and hidden dimensions for all models.
- **Noise**: Models were trained with Gaussian noise (std=0.2) and evaluated on clean and noisy (std=0.5) test sets.

## Results

| Mode | Clean Test Acc | Noisy Test Acc (std=0.5) |
|------|----------------|--------------------------|
| None | 75.75%         | 58.35%                   |
| ZCA  | 82.20%         | 51.80%                   |
| Soft-PCA | 83.00%     | 52.60%                   |
| **SGW**  | 79.10%     | **60.20%**               |

## Findings
1. **Adaptive Robustness**: SGW achieved the highest robustness to heavy test-time noise (60.20%), outperforming both the unwhitened baseline and the standard ZCA.
2. **Learned Filtering**: Analysis of the learned gate $s(\lambda)$ revealed that SGW automatically learned to suppress components with small eigenvalues (gate values ~0.04-0.09) and emphasize those with large eigenvalues (gate values ~0.55-0.68). This behavior mirrors soft-thresholding but is discovered purely from task gradients.
3. **Whitening Benefit**: Interestingly, while whitening (ZCA/Soft-PCA) improved clean accuracy significantly over the unwhitened baseline (83% vs 75%), it made the models more sensitive to noise. SGW found a middle ground, offering better clean performance than the baseline while providing the best noise robustness.

## Plots
The following plots are available in the experiment directory:
- `results.png`: Bar chart comparing clean and noisy accuracies across all modes.
- `sgw_gate.png`: Visualization of the learned spectral gate as a function of eigenvalues (log scale).
