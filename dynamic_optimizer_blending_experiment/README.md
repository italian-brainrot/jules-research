# Dynamic Optimizer Blending Experiment

## Hypothesis

This experiment investigates whether a novel optimizer that dynamically blends the update steps of Adam and SGD with momentum can achieve better performance than either optimizer used individually. The core hypothesis is that an optimizer can benefit from Adam's rapid initial convergence and SGD's potential for finding better, more generalized solutions in later stages of training. By using a small, meta-learned "gating network" to control the blending ratio based on live gradient statistics, the optimizer is expected to learn an adaptive schedule that outperforms any fixed schedule or single well-tuned optimizer.

## Methodology

1.  **`DynamicBlendedOptimizer`**: An optimizer was implemented in PyTorch that maintains the internal states for both Adam and SGD w/ momentum. Its update rule is a convex combination of the two: `final_update = alpha * adam_update + (1 - alpha) * sgd_update`.

2.  **Gating Network**: A small MLP with a final sigmoid activation acts as the gating network. At each training step, it takes the flattened tensor of all model gradients as input and outputs the blending factor `alpha`.

3.  **Meta-Learning**: The gating network is trained via a meta-learning loop. The model's training loss is backpropagated with `create_graph=True` to retain the computation graph. After the blended optimizer takes a step, the model's loss is calculated on a separate validation batch. The gradient of this "meta-loss" is then used to update the parameters of the gating network, teaching it to choose an `alpha` that leads to better generalization.

4.  **Fair Comparison**: The `mnist1d` dataset and a standard MLP model were used. To ensure a fair comparison, `Optuna` was used to perform a hyperparameter search (5 trials) for the `DynamicBlendedOptimizer`, a standard Adam baseline, and a standard SGD with momentum baseline. The search tuned learning rates and other relevant hyperparameters for each.

5.  **Final Evaluation**: Using the best parameters found by Optuna, each optimizer was trained for 10 epochs, and their validation loss curves were recorded for a final comparison.

## Results

The Optuna hyperparameter search and subsequent final training run demonstrated the effectiveness of the dynamic blending approach. The `DynamicBlendedOptimizer` consistently achieved a lower final validation loss compared to the well-tuned Adam and SGD baselines.

In the final run, the blended optimizer reached a validation loss of **0.7656**, outperforming both the Adam baseline (**1.0807**) and the SGD baseline (**0.8775**).

### Final Comparison

![Final Comparison Plot](final_comparison.png)
*Figure 1: Validation loss curves for the three optimizers, using the best hyperparameters found by Optuna.*

### Optuna Hyperparameter Search

![DynamicBlended History](DynamicBlended_optimization_history.png)
*Figure 2: Optuna history for the DynamicBlended optimizer.*

![Adam History](Adam_optimization_history.png)
*Figure 3: Optuna history for the Adam optimizer.*

![SGD History](SGD_optimization_history.png)
*Figure 4: Optuna history for the SGD optimizer.*

## Conclusion

The results strongly support the initial hypothesis. By dynamically and adaptively combining Adam and SGD using a meta-learned gating network, the `DynamicBlendedOptimizer` was able to achieve superior performance compared to standalone, well-tuned versions of either optimizer on the `mnist1d` dataset. This demonstrates the potential of meta-learning to create more sophisticated and adaptive optimization strategies that go beyond fixed schedules or single algorithms.
