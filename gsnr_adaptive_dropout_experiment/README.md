# GSNR-Adaptive Dropout (GAD) Experiment

## Hypothesis
The Gradient Signal-to-Noise Ratio (GSNR) measures the consistency of gradients during training. We hypothesize that scaling the dropout probability $p$ by $(1 - GSNR)^\gamma$ improves generalization.

Specifically:
- Early in training, gradients are usually consistent (high GSNR), so dropout is reduced to allow the model to find a good direction quickly.
- As training progresses and the model reaches noisier regions or starts fitting fine-grained details, GSNR decreases, and dropout increases to provide stronger regularization.

## Implementation
- **GSNR Estimation**: GSNR is estimated per-layer using the moving averages of gradients (m) and squared gradients (v) from the Adam optimizer: $GSNR = \frac{\hat{m}^2}{\hat{v} + \epsilon}$.
- **Adaptive Dropout**: The effective dropout rate for each layer is $p_{eff} = p_{base} \cdot (1 - GSNR_{layer})^\gamma$.
- **Experiment**: We compare GAD against a baseline MLP with constant dropout on the `mnist1d` dataset. Both models are tuned using Optuna (30 trials each) for learning rate, weight decay, and dropout parameters.

## Results
The experiment was conducted on `mnist1d` with 8000 samples.

### Baseline (Constant Dropout)
- **Best Params**: `{'lr': 0.00968, 'weight_decay': 0.00046, 'p': 0.133}`
- **Test Accuracy**: 0.7775

### GAD (GSNR-Adaptive Dropout)
- **Best Params**: `{'lr': 0.00389, 'weight_decay': 0.00022, 'p_base': 0.206, 'gamma': 2.06}`
- **Test Accuracy**: 0.7556

## Conclusion
In this specific experiment on `mnist1d`, GAD achieved lower training loss compared to the baseline but showed slightly lower test accuracy. This suggests that while GAD improves optimization efficiency (faster fitting of training data), the dynamic curriculum it creates (low-to-high dropout) may lead to more overfitting than a constant dropout rate on this small-scale dataset.

Further research could explore GAD on larger datasets where optimization is more challenging, or different functions for mapping GSNR to dropout probability.

## Visualizations
- `results.png`: Training loss and validation accuracy curves.
- `gsnr_check.png`: Plot showing the behavior of GSNR estimation during training.
