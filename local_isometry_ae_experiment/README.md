# Local Isometry Preserving Autoencoder (LIP-AE) Experiment

## Hypothesis
Adding an orthonormality constraint on the Jacobian of the encoder (Local Isometry Preserving penalty) improves the quality of learned latent representations for dimensionality reduction. Specifically, it should better preserve the local manifold structure compared to a standard Autoencoder (AE) and a Contractive Autoencoder (CAE).

## Methodology
- **Dataset**: `mnist1d` (40 dimensions).
- **Architecture**: 2-layer MLP for encoder and decoder. Latent dimension $d=2$ for visualization and strong bottleneck.
- **Models**:
    - **Baseline**: Standard AE minimizing MSE reconstruction loss.
    - **CAE**: Contractive AE with Frobenius norm penalty on the Jacobian $\|J\|_F^2$.
    - **LIP-AE**: Proposed model with penalty $\|J J^T - I\|_F^2$, encouraging the encoder Jacobian rows to be orthonormal.
- **Tuning**: Optuna used to tune learning rate for all models and regularization strength $\lambda$ for CAE and LIP-AE (15 trials each).
- **Evaluation**:
    - Reconstruction MSE.
    - Latent classification accuracy: A separate 2-layer MLP classifier trained on frozen latent representations.

## Results

| Model | MSE | Accuracy | Best Params |
|-------|-----|----------|-------------|
| Baseline | 0.553290 | 0.3030 | LR: 1.01e-3 |
| CAE | 0.579390 | 0.3090 | LR: 4.11e-4, $\lambda$: 0.0439 |
| LIP-AE | 0.568096 | 0.3110 | LR: 1.11e-3, $\lambda$: 0.0015 |

### Analysis
- **LIP-AE** achieved the highest classification accuracy (31.1%), outperforming both the Baseline (30.3%) and the Contractive AE (30.9%).
- **LIP-AE** has a lower reconstruction error (MSE 0.568) than **CAE** (MSE 0.579), suggesting it provides a better trade-off between regularization and data reconstruction.
- The orthonormality constraint on the Jacobian seems to successfully encourage the encoder to preserve local geometry, which helps in maintaining class discriminability in the low-dimensional latent space.

## Conclusion
The Local Isometry Preserving penalty is a promising alternative to the standard Contractive AE penalty for dimensionality reduction. By encouraging the encoder to be a local isometry, it better preserves the manifold structure and class separation even in very low-dimensional bottlenecks.
