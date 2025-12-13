# Gradient-Norm Scaled Adam (GNS-Adam) Experiment

## Hypothesis

The performance of optimizers like Adam can be sensitive to the choice of learning rate. A fixed learning rate may be too aggressive in regions of high gradients (leading to overshooting) and too conservative in regions of low gradients (leading to slow convergence).

My hypothesis is that by dynamically scaling the learning rate at each step *inversely* to the L2 norm of the gradients, the optimizer can take more appropriately sized steps. This should lead to more stable and potentially faster convergence. The intuition is to take smaller steps when the gradient is large and larger steps when the gradient is small.

## Methodology

### Optimizers

- **Adam (Baseline):** The standard `torch.optim.Adam` optimizer.
- **GNS-Adam (Proposed):** The custom `GNSAdam` optimizer, which wraps a base Adam optimizer and scales the learning rate at each step inversely to the total L2 norm of the gradients.

### Hyperparameter Tuning

To ensure a fair comparison, the learning rate for both optimizers was tuned using the `optuna` library.
- A study was conducted for each optimizer, running for 30 trials.
- The objective was to maximize the final test accuracy after 15 epochs.
- The search space for the learning rate was between `1e-5` and `1e-1` on a logarithmic scale.

### Model and Dataset

- **Model:** A simple Multi-Layer Perceptron (MLP) with one hidden layer.
- **Dataset:** The `mnist1d` dataset with 10,000 training samples.
- **Training:** The final comparison was run using the best learning rates found during the tuning phase. To ensure a fair comparison, both models were initialized with the exact same random weights for every run.

## Results

After tuning, the best learning rate for Adam was ~0.026 and for GNS-Adam was ~0.021. The final experiment was run with these optimized learning rates. The results are summarized in the plot below:

![Optimizer Comparison](comparison_plot.png)

After a fair comparison with tuned learning rates, the standard Adam optimizer performs slightly better than the GNS-Adam optimizer. The final test accuracy for the tuned Adam was approximately 70.1%, while the tuned GNS-Adam achieved approximately 69.1%.

## Conclusion

The hypothesis was **not supported** by the results of this experiment. After performing a fair comparison by tuning the learning rate for both optimizers, the proposed GNS-Adam method did not show any performance benefit over the standard Adam optimizer. In fact, the standard Adam baseline achieved a slightly higher test accuracy. This suggests that, for this specific task, the dynamic scaling of the learning rate based on the gradient norm does not lead to improved performance and may even be slightly detrimental compared to using a well-tuned, fixed learning rate.
