# Proximal Gradient Optimizer Experiment

## Hypothesis

This experiment investigates whether incorporating a proximal operator into a standard optimizer (Adam) can improve model performance. The central hypothesis is that applying a proximal step, specifically soft-thresholding for L1 regularization, after the main gradient update can lead to better generalization and a lower final validation loss compared to a standard, well-tuned Adam optimizer.

## Methodology

To ensure a fair and scientifically valid comparison, the following methodology was employed:

1.  **Dataset**: The `mnist1d` dataset was used for both training and validation. The data was normalized before being fed into the model.

2.  **Model**: A simple multi-layer perceptron (MLP) with two hidden layers and ReLU activation functions was used as the base model for all experiments.

3.  **Optimizers**:
    *   **Baseline**: The standard `torch.optim.Adam` optimizer.
    *   **Proposed**: A custom `ProximalOptimizer` was implemented, which wraps a base optimizer (Adam in this case) and applies a proximal operator—soft-thresholding—to the model's parameters after each step.

4.  **Hyperparameter Tuning**: To eliminate learning rate as a confounding variable, [Optuna](https://optuna.org/) was used to independently tune the learning rate for both the baseline and the proposed optimizer. Each optimizer was subjected to 30 trials to find the learning rate that minimized the validation loss.

5.  **Final Evaluation**: After identifying the optimal learning rates, both optimizers were used to train the model for an extended run of 50 epochs. The final validation loss from this run was used as the primary metric for comparison.

## Results

The experiment was executed using the `compare.py` script. The results were as follows:

*   **Optimal Learning Rate (Adam)**: `0.003640`
*   **Optimal Learning Rate (ProximalAdam)**: `0.005657`

After training for 50 epochs with these optimal learning rates, the final validation losses were:

*   **Final Validation Loss (Adam)**: `1.6212`
*   **Final Validation Loss (ProximalAdam)**: `1.8358`

## Conclusion

The results of this experiment **do not support the initial hypothesis**. The standard Adam optimizer, when its learning rate was properly tuned, achieved a significantly lower validation loss than the `ProximalAdam` optimizer.

The introduction of the soft-thresholding operator, even when its strength was tied to a tuned learning rate, did not lead to improved performance on this task. In fact, it appears to have had a detrimental effect, hindering the optimizer's ability to find a good minimum. This suggests that for this particular model and dataset, the implicit regularization provided by the proximal operator was not beneficial and may have been too aggressive, preventing the model from fitting the data effectively.
