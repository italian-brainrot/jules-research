# Adaptive Weight Decay Experiment

This experiment investigates a novel regularization method called "Adaptive Weight Decay," where the strength of the L2 regularization penalty is proportional to the magnitude of each weight.

## Hypothesis

The hypothesis is that making the weight decay coefficient for each parameter proportional to its magnitude would provide a more effective form of regularization than standard L2 weight decay, which uses a single, fixed coefficient for all parameters. The intuition is that larger weights should be penalized more strongly to prevent them from dominating the network.

## Methodology

To test this hypothesis, a simple Multi-Layer Perceptron (MLP) was trained on the `mnist1d` dataset. The experiment compared two methods:

1.  **Baseline:** A standard Adam optimizer with a fixed L2 weight decay coefficient.
2.  **New Method:** An Adam optimizer where the gradients were manually adjusted before the `optimizer.step()` call to apply the adaptive weight decay penalty. The penalty for each parameter was calculated as `strength * |w| * w`, where `w` is the parameter's weight.

For a fair comparison, `optuna` was used to perform a hyperparameter search (50 trials each) to find the optimal learning rate and regularization strength for both methods. The metric for comparison was the final validation loss.

## Results

After running the `optuna` studies, the best results for each method were as follows:

| Method                          | Best Validation Loss | Best Learning Rate | Best Regularization Strength |
| ------------------------------- | -------------------- | ------------------ | ---------------------------- |
| **Baseline (Adam + L2)**        | **1.0491**           | `0.00476`          | `0.000056` (weight_decay)    |
| **Adaptive Weight Decay**       | 1.0534               | `0.00388`          | `0.00039` (adaptive_wd_strength) |

## Conclusion

The experimental results show that the standard Adam optimizer with a tuned, fixed weight decay coefficient achieved a slightly lower validation loss than the proposed Adaptive Weight Decay method.

Therefore, the initial hypothesis was **not supported**. On the `mnist1d` dataset, making the weight decay penalty proportional to the weight's magnitude did not lead to improved generalization and performed slightly worse than the standard, well-established L2 regularization method.
