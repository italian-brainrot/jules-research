# Polynomial Regularization Experiment

## Hypothesis

Adding a regularization term proportional to the squared norm of the change in weights (the update step) can improve generalization. This regularization penalizes large, abrupt changes in the weight space, which is hypothesized to lead to a smoother and more generalizable model. The regularization term is defined as: `reg_strength * ||Wt - Wt-1||^2`, where `Wt` are the weights at the current step and `Wt-1` are the weights from the previous step.

## Methodology

A simple MLP model was trained on the `mnist1d` dataset. Two training regimes were compared:

1.  **Polynomial Regularization:** An Adam optimizer with the addition of the polynomial regularization term to the loss function.
2.  **Baseline:** A standard Adam optimizer.

Optuna was used to perform a hyperparameter search for both methods to ensure a fair comparison. The search space for the Polynomial Regularization method included learning rate and regularization strength, while the search for the Adam baseline focused on the learning rate.

## Results

After running the Optuna study for 10 trials each with the corrected implementation, the following results were obtained:

*   **Polynomial Regularization:**
    *   Best Accuracy: `0.736`
    *   Best Parameters:
        *   `lr`: `0.0177`
        *   `reg_strength`: `0.0043`

*   **Adam Baseline:**
    *   Best Accuracy: `0.7115`
    *   Best Parameters:
        *   `lr`: `0.0045`

## Conclusion

The results from the corrected experiment show that the Polynomial Regularization method slightly outperformed the standard Adam optimizer on this task. The Polynomial Regularization method achieved a higher accuracy (0.736) compared to the Adam baseline (0.7115). This suggests that penalizing large changes in the weight space can indeed lead to improved generalization, supporting the initial hypothesis.
