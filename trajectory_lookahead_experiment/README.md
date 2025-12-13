# Lookahead Optimizer with Trajectory Averaging

## Hypothesis

The standard Lookahead optimizer updates its "slow" weights by interpolating between the previous slow weights and the current "fast" weights. My proposed modification is to update the slow weights by *averaging the model parameters from several points along the recent trajectory of the fast weights*. This is inspired by Polyak-Ruppert averaging. The hypothesis is that this will find a more robust minimum in the loss landscape, leading to better generalization.

## Methodology

### Optimizers

To ensure a fair comparison, the learning rates for both optimizers were tuned using the Optuna hyperparameter optimization framework. For each optimizer, a TPE sampler was used to search for the best learning rate over 20 trials in the range `[1e-5, 1e-1]`.

- **Adam:** A standard Adam optimizer.
- **Trajectory Lookahead:** The proposed optimizer, wrapping an Adam optimizer. The lookahead parameters were `la_steps=5` and `trajectory_len=3`.

### Model and Dataset

- **Model:** A simple Multi-Layer Perceptron (MLP) with one hidden layer.
- **Dataset:** The `mnist1d` dataset with 10,000 training samples.
- **Training:** Both models were trained for 5 epochs, starting from the same randomly initialized weights to ensure a fair comparison.

## Results

The experiment was run after tuning the learning rates, and the results are summarized in the plot below:

![Optimizer Comparison](comparison_plot.png)

After tuning, the performance of the two optimizers is much more comparable. The Trajectory Lookahead optimizer's performance is now very close to that of the standard Adam optimizer.

## Conclusion

The initial hypothesis was that trajectory averaging would lead to better generalization. After conducting a fairer comparison with tuned learning rates, the results show that the Trajectory Lookahead optimizer performs comparably to the standard Adam optimizer. While it did not significantly outperform the baseline, it is no longer clearly detrimental, as suggested by the initial results. This suggests that with proper tuning, trajectory averaging can be a viable, if not superior, optimization strategy.
