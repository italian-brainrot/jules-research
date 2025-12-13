# Gradient Polynomial Extrapolation (GPE) Optimizer Experiment

This experiment investigates a novel optimizer called Gradient Polynomial Extrapolation (GPE), which aims to accelerate convergence by fitting a polynomial to the recent trajectory of each parameter's gradient and extrapolating a future gradient.

## Hypothesis

The core hypothesis is that the trajectory of gradients is more stable and predictable than the trajectory of the parameters themselves. By extrapolating the gradient, we can "denoise" the updates and provide a better-informed update direction to a base optimizer (like Adam), leading to faster and more stable convergence.

## Methodology

1.  **Optimizer Implementation**: A PyTorch optimizer named `GPE` was implemented in `optimizer.py`. This optimizer wraps a base optimizer (e.g., `torch.optim.Adam`) and maintains a history of gradients. In each step, it fits a 2nd-degree polynomial to the last 10 gradients for each parameter, extrapolates the next gradient, and then uses a weighted average of the current and extrapolated gradients (`alpha=0.4`) to update the model.

2.  **Comparison Setup**: The `compare.py` script was created to benchmark the performance of `GPE(Adam)` against the standard `Adam` optimizer.
    *   **Dataset**: The `mnist1d` dataset was used, with 10,000 training samples.
    *   **Model**: A simple Multi-Layer Perceptron (MLP) with one hidden layer of 128 neurons and a ReLU activation function.
    *   **Fairness**: To ensure a fair comparison, both optimizers started with the exact same initial model weights. Crucially, the learning rate, a key hyperparameter, was tuned for each optimizer independently using Optuna.

3.  **Learning Rate Tuning**:
    *   An objective function was created to train the model for 20 epochs and return the best validation loss.
    *   Optuna was used to run 20 trials for both Adam and GPE(Adam) to find the learning rate that minimized the validation loss.

4.  **Execution**: After finding the optimal learning rates, the script trained the model for 50 epochs with both optimizers using their respective best learning rates and recorded the validation loss at each epoch.

## Results

After learning rate tuning, the optimal learning rate for Adam was found to be ~0.034 and for GPE(Adam) was ~0.072. The final comparison using these learning rates showed that both optimizers achieved a similar validation loss, with Adam performing slightly better. The performance is visualized in the plot below:

![Comparison Plot](comparison_plot.png)

As seen in the plot, the performance of the `GPE(Adam)` optimizer is still very similar to the standard `Adam` optimizer, even after both have been tuned.

## Conclusion

The initial hypothesis that extrapolating gradients would lead to faster or more stable convergence is not supported by the results, even after a more rigorous, fair comparison involving learning rate tuning. The `GPE` optimizer performed nearly identically to the standard Adam baseline. The added computational overhead of GPE, combined with its lack of performance benefit, makes it a less efficient choice.
