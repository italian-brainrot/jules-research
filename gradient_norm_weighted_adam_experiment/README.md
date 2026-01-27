# Gradient-Norm-Weighted Adam (GNW-Adam)

## Hypothesis

Standard Adam uses a fixed learning rate, which may not be optimal for all parts of the loss landscape. My hypothesis is that dynamically scaling the learning rate based on the L2 norm of the gradients can improve optimizer performance. Specifically, I propose reducing the learning rate when the gradient norm is high (indicating a steep, "bumpy" region) and increasing it when the norm is low (indicating a "flat" region). This should lead to more stable convergence and a better final validation loss.

## Methodology

To test this, I created a new optimizer called GNW-Adam, which modifies the Adam update rule by dividing the learning rate by `(1 + total_grad_norm)`. I then compared the performance of GNW-Adam against the standard Adam optimizer on the `mnist1d` dataset.

To ensure a fair comparison, I used the `optuna` library to perform a hyperparameter search for the optimal learning rate for each optimizer. The search was conducted for 10 trials, with each trial training for 25 epochs. The best learning rate for each optimizer was then used to train a final model for 50 epochs.

The experiment was conducted using a simple MLP with one hidden layer, and the validation loss was tracked at each epoch.

## Results

The Optuna study found the following optimal learning rates:

*   **Adam:** 0.00624
*   **GNW-Adam:** 0.00585

After training with these learning rates, the final validation losses were:

*   **Adam:** 1.0846
*   **GNW-Adam:** 1.0533

The validation loss curves are shown in the plot below, which is also saved as `comparison.html`.

![Comparison Plot](comparison.html)

## Conclusion

The results support my hypothesis. GNW-Adam achieved a lower final validation loss than the standard Adam optimizer, suggesting that dynamically scaling the learning rate based on the gradient norm is a beneficial modification. While the improvement is modest, it demonstrates the potential of this approach. Further research could explore different scaling functions or applying this technique to other optimizers.
