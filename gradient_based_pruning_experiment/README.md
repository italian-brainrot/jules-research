# Gradient-based Pruning Experiment

This experiment investigates a pruning technique based on the magnitude of gradients.

## Hypothesis

The hypothesis is that weights with consistently small gradient magnitudes are less important for the learning process and can be pruned without significantly impacting performance. This is in contrast to the more common magnitude-based pruning, where weights with small values are removed. The rationale is that a small gradient indicates that the weight is not contributing much to the reduction of the loss, and is therefore a good candidate for pruning.

## Methods

A simple Multi-Layer Perceptron (MLP) was trained on the `mnist1d` dataset. Two pruning methods were compared:

1.  **Gradient-based Pruning:** The moving average of the absolute value of each weight's gradient was tracked during training. After a warm-up period of 10 epochs, a certain percentage of the weights with the lowest average gradient magnitudes were pruned.
2.  **Magnitude-based Pruning (Baseline):** As a baseline, standard magnitude-based pruning was implemented. After 10 epochs, a certain percentage of the weights with the lowest magnitudes were pruned.

Both methods were evaluated across a range of sparsity levels, from 10% to 90%.

## Results

The following plot shows the test accuracy of the models pruned with each method at different sparsity levels.

![Pruning Method Comparison](pruning_comparison.png)

The results show that magnitude-based pruning consistently outperforms gradient-based pruning across all tested sparsity levels. The performance of the gradient-based pruning method degrades more quickly as the sparsity level increases.

## Conclusion

The hypothesis that pruning weights with small gradient magnitudes is an effective pruning strategy is not supported by the results of this experiment. Magnitude-based pruning appears to be a more effective method for maintaining accuracy at various levels of sparsity. It's possible that the moving average of the gradient is not a good indicator of a weight's importance, or that the hyperparameters of the gradient-based method (such as the momentum of the moving average) were not optimally tuned. Further research could explore different ways of accumulating gradient information to identify prunable weights.
