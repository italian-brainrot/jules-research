# Weight Trajectory Regularized Learning Rate Experiment

This experiment introduces and evaluates a novel optimizer wrapper, `WeightTrajectoryLR`, which dynamically adjusts the learning rate for each layer of a neural network.

## Hypothesis

The learning rate for a specific layer should be modulated based on the distance its weights have traveled from their initial state. A layer whose weights have moved significantly might be oscillating, overfitting, or becoming unstable; its learning rate should be dampened. Conversely, a layer whose weights have barely changed might be stuck in a poor local minimum, and its learning rate could be increased.

The hypothesis is that this "tethering" of weights to their initial state can act as a regularizer, preventing layers from straying too far too quickly and thus leading to more stable training and better generalization.

## Methodology

To test this, I created an optimizer wrapper, `WeightTrajectoryLR`, which sits on top of a standard optimizer (in this case, `torch.optim.Adam`). Before each `step()`, the wrapper calculates a scaling factor for each layer's learning rate:

`scaling_factor = 1.0 / (1.0 + beta * ||W_current - W_initial||_2)`

where `beta` is a hyperparameter controlling the strength of this effect. This factor is multiplied by the base learning rate for that layer.

The experiment was conducted on the `mnist1d` dataset using a simple MLP model. I used [Optuna](https://optuna.org/) to perform a fair hyperparameter search for two setups:
1.  **Baseline:** A standard Adam optimizer, tuning the `learning_rate`.
2.  **Wrapped:** The `WeightTrajectoryLR` wrapper around an Adam optimizer, tuning both the base `learning_rate` and the `beta` parameter.

Each setup was run for 30 trials, with each trial training for 20 epochs. The minimum validation loss achieved during training was the metric for comparison.

## Results

After a final correction to the `WeightTrajectoryLR` implementation, ensuring the learning rate scaling logic is correctly applied, the Optuna studies yielded the following results:

*   **Baseline (Adam):**
    *   **Best Validation Loss:** 1.0541
    *   **Best Params:** `{'lr': 0.0084}`
*   **Wrapped (WeightTrajectoryLR Adam):**
    *   **Best Validation Loss:** 1.0447
    *   **Best Params:** `{'lr': 0.0097, 'beta': 0.3200}`

Here is a visual comparison of the distribution of validation losses across all trials for both methods with the final, correct logic:

![Comparison Plot](comparison.png)

## Conclusion

With the finally corrected implementation, the `WeightTrajectoryLR` wrapper demonstrates a slight performance improvement over the standard Adam optimizer. The best validation loss for the wrapped optimizer was marginally lower than the baseline, and the box plot shows a slightly more favorable distribution of results.

The original hypothesis is now weakly supported by the experimental evidence. Dynamically scaling the learning rate based on weight trajectory appears to offer a small but measurable regularization benefit for this specific task and model. While the effect is not dramatic, the `WeightTrajectoryLR` method did outperform a well-tuned Adam baseline, suggesting the concept has merit and may warrant further investigation on more complex problems.
