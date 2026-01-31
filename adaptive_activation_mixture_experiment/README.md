# Adaptive Activation Mixture Experiment

## Hypothesis
We hypothesize that a neural network can benefit from using a mixture of different activation functions (ReLU, Tanh, Sin, and Identity) where the mixing weights are learned during training. Different layers might prefer different activations based on the depth and the nature of the features they are learning. Specifically, on the MNIST-1D dataset, which contains 1D signals, periodic activations like Sin might be particularly useful.

## Methodology
- **Model**: A 3-layer MLP (40 -> 256 -> 256 -> 10).
- **Activations**:
    - **Baseline**: Standard ReLU activation.
    - **AMA (Adaptive Mixture of Activations)**: A learnable mixture of ReLU, Tanh, Sin(omega * x), and Identity.
- **Learnable Parameters**:
    - Mixing weights (via Softmax).
    - Frequency `omega` for the Sin activation (initialized to 1.0).
- **Training**:
    - Dataset: MNIST-1D with 10,000 samples.
    - Hyperparameter Tuning: Learning rate tuned using Optuna (15 trials) for both models.
    - Final Evaluation: 30 epochs, averaged over 3 seeds.

## Results
The experiment yielded the following results:

| Metric | Baseline (ReLU) | Adaptive Mixture (AMA) |
|--------|-----------------|------------------------|
| Mean Accuracy | 0.7718 | 0.7590 |
| Std Dev | 0.0041 | 0.0036 |
| Best Learning Rate | 0.0085 | 0.0035 |

### Learned Activation Weights
The final mixing weights for the AMA model showed a strong preference for the **Sin** activation:

- **Layer 1**: ReLU (24.7%), Tanh (7.9%), **Sin (61.5%)**, Identity (5.8%)
- **Layer 2**: ReLU (3.3%), Tanh (1.7%), **Sin (92.4%)**, Identity (2.7%)

The frequency `omega` for the Sin activation evolved to **2.16** in Layer 1 and **1.29** in Layer 2.

### Analysis
1. **Sin Preference**: Interestingly, the model overwhelmingly chose the Sin activation, especially in the deeper layer. This suggests that for the 1D signal data in MNIST-1D, periodic non-linearities are highly expressive.
2. **Performance**: Despite the strong preference for Sin, the AMA model slightly underperformed the ReLU baseline (75.9% vs 77.2%). This might be due to the increased optimization complexity of having learnable activation parameters or the potential for instability with Sin activations during the early stages of training.
3. **Layer Differences**: Layer 1 retained a significant amount of ReLU (24.7%), while Layer 2 became almost entirely Sin-based. This suggests that the initial layer benefits from the sparsity-inducing properties of ReLU, while subsequent layers benefit from the smooth, periodic nature of Sin.

## Plots
The following plots are available in the experiment directory:
- `comparison.png`: Shows the accuracy and loss curves for both models.
- `mixture_weights_evolution.png`: Shows how the mixing weights changed over time for each layer.
- `omegas_evolution.png`: Shows the evolution of the Sin frequency parameter.

## Conclusion
The experiment demonstrates that neural networks, when given the choice, can favor non-standard activations like Sin over ReLU for certain types of data. While the adaptive mixture did not outperform the tuned ReLU baseline in this specific setup, the strong preference for Sin activations opens up questions about whether specialized architectures (like SIRENs or mixtures) might be better suited for 1D signal processing tasks than standard ReLU-based MLPs.
