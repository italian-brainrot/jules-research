# Gradient Coherence Optimizer Experiment

## Hypothesis
My hypothesis is that by dynamically scaling each parameter's update by the historical consistency of its gradient's sign, the optimizer can accelerate learning for parameters with stable gradients while dampening oscillations for those with noisy or conflicting gradients, leading to improved performance over standard Adam.

## Method
To test this hypothesis, I designed and implemented a novel optimizer, which I named `GradientCoherence`. The core mechanism of this optimizer is to calculate a "coherence factor" for each parameter, which is defined as the ratio of the exponential moving average of the gradient to the exponential moving average of its absolute value. This factor is then used to modulate the learning rate for each parameter individually, allowing the optimizer to dynamically adjust the update step based on the stability of the gradient's direction.

To ensure a fair and robust comparison, I conducted a hyperparameter search for the learning rate of both the `GradientCoherence` optimizer and the standard `Adam` optimizer using the `optuna` library. Both optimizers were tasked with training a simple multi-layer perceptron on the `mnist1d` dataset. I ran 50 trials for each optimizer to find the learning rate that yielded the best validation loss.

## Results
After completing the hyperparameter search, the results were as follows:

- **Adam Optimizer:**
  - Best Validation Loss: `0.9963`
  - Best Learning Rate: `0.003891`

- **Gradient Coherence Optimizer:**
  - Best Validation Loss: `1.0072`
  - Best Learning Rate: `0.095112`

## Conclusion
The results of the experiment show that the standard `Adam` optimizer achieved a slightly lower validation loss than the newly proposed `GradientCoherence` optimizer. This outcome does not support the initial hypothesis, as the method of scaling parameter updates by the coherence of their gradients did not lead to better performance on this specific task.

While the `GradientCoherence` optimizer did not outperform `Adam`, the experiment serves as a valuable exploration into adaptive learning rate mechanisms. It is possible that this method could prove more effective in different contexts, such as with deeper or more complex model architectures, or on datasets with different characteristics. However, for this particular problem, the added complexity of the coherence calculation did not translate into a tangible improvement in performance.

A potential area for future work would be to incorporate bias correction for the moving averages, similar to the technique used in the Adam optimizer. The absence of this correction in the current implementation might have contributed to the suboptimal performance, and adding it could lead to a more competitive optimizer.
