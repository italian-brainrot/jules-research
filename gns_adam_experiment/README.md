# Gradient-Norm Scaled Adam (GNS-Adam) Experiment

## Hypothesis

The performance of optimizers like Adam can be sensitive to the choice of learning rate. A fixed learning rate may be too aggressive in regions of high gradients (leading to overshooting) and too conservative in regions of low gradients (leading to slow convergence).

My hypothesis is that by dynamically scaling the learning rate at each step *inversely* to the L2 norm of the gradients, the optimizer can take more appropriately sized steps. This should lead to more stable and potentially faster convergence. The intuition is to take smaller steps when the gradient is large and larger steps when the gradient is small.

## Methodology

### Optimizers

- **Adam (Baseline):** The standard `torch.optim.Adam` optimizer was used as a baseline for comparison. A learning rate of `0.001` was used.
- **GNS-Adam (Proposed):** The custom `GNSAdam` optimizer was implemented. This optimizer wraps a base Adam optimizer. In each `step`, it calculates the total L2 norm of the gradients across all model parameters. It then scales the learning rate for that step by dividing the initial learning rate by this total gradient norm. The same base learning rate of `0.001` was used.

### Model and Dataset

- **Model:** A simple Multi-Layer Perceptron (MLP) with one hidden layer.
- **Dataset:** The `mnist1d` dataset with 10,000 training samples was used.
- **Training:** To ensure a fair comparison, both models were initialized with the exact same random weights. Both were trained for 15 epochs.

## Results

The experiment was run with the corrected optimizer logic, comparing the test accuracy of the two optimizers at the end of each epoch. The results are summarized in the plot below:

![Optimizer Comparison](comparison_plot.png)

With the corrected logic, the performance of the two optimizers is much closer. The GNS-Adam optimizer shows a slight advantage, performing comparably to or slightly better than the standard Adam optimizer throughout the training process. It ultimately achieves a slightly higher final test accuracy after 15 epochs (approximately 62% for GNS-Adam vs. 61% for Adam).

## Conclusion

The hypothesis was **weakly supported** by the results of this experiment. While the proposed GNS-Adam optimizer did not show a dramatic improvement, it consistently performed on par with or slightly better than the standard Adam optimizer. This suggests that the method is not detrimental and could potentially offer a small benefit. The strategy of scaling the learning rate inversely to the gradient norm appears to be a viable, if not overwhelmingly superior, approach. Further testing on different datasets and model architectures would be needed to determine if this method provides a more significant advantage in other contexts.
