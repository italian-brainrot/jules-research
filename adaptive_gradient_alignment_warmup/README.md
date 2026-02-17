# Adaptive Gradient Alignment Warmup (AGAW)

## Hypothesis
A learning rate warmup schedule that adapts to the stability of the gradient direction can reach the target learning rate more effectively than a fixed linear warmup. Specifically, we hypothesize that by increasing the learning rate proportionally to the cosine similarity between consecutive gradient updates, the model can safely accelerate training in stable regions and remain cautious in chaotic regions.

## Methodology
To test this hypothesis, we implemented the `AGAWOptimizer`, a wrapper for standard optimizers (e.g., Adam).

### AGAW Algorithm
At each training step $t$:
1. Compute the current gradient $g_t$.
2. Compute the cosine similarity $s_t$ between $g_t$ and the gradient from the previous step $g_{t-1}$.
3. Update the current learning rate $\eta_t$:
   $$\eta_t = \eta_{t-1} + \frac{\eta_{target} - \eta_{initial}}{W} \cdot \max(0, s_t)^\gamma$$
   where:
   - $\eta_{target}$ is the desired peak learning rate.
   - $W$ is a nominal warmup period (number of steps).
   - $\gamma$ is a sensitivity parameter.
4. Cap $\eta_t$ at $\eta_{target}$.

### Experimental Setup
- **Dataset**: MNIST-1D (5000 samples).
- **Model**: 4-layer MLP (Input: 40, Hidden: 256, Output: 10).
- **Comparison**:
  1. **Baseline**: Standard Adam without warmup.
  2. **Linear Warmup**: Adam with a fixed linear warmup schedule.
  3. **AGAW**: Adam with the proposed adaptive warmup.
- **Hyperparameter Tuning**: Optuna was used to tune the learning rate and method-specific parameters for each mode (15 trials per mode).
- **Final Evaluation**: Each method was trained for 50 epochs using its best hyperparameters.

## Results

| Mode | Best Test Accuracy | Best Hyperparameters |
| :--- | :---: | :--- |
| **Baseline** | 0.6800 | `lr`: 0.0047 |
| **Linear Warmup** | 0.6750 | `lr`: 0.0049, `warmup_steps`: 159 |
| **AGAW** | **0.6890** | `lr`: 0.0046, `gamma`: 1.70, `warmup_steps_nominal`: 107 |

### Observations
- AGAW achieved the highest test accuracy (68.90%), outperforming both the standard Adam baseline and the fixed linear warmup.
- The optimal `gamma` for AGAW was around 1.7, suggesting that a non-linear dependence on gradient alignment is beneficial.
- AGAW's adaptive nature allows it to "find its own way" to the target learning rate, potentially avoiding unstable updates in the early stages of training better than a fixed-time schedule.

## Conclusion
The results support the hypothesis that adaptive warmup based on gradient alignment can improve model performance. By monitoring the consistency of the gradient signal, AGAW provides a more principled way to ramp up the learning rate compared to traditional time-based schedules. This approach is particularly useful when the initial loss landscape is highly variable or when the optimal warmup period is unknown.

Future work could explore per-layer adaptive warmup or combining AGAW with other adaptive learning rate methods.
