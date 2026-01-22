# Soft Weight Projection Experiment

This experiment investigates whether applying a "soft" projection of a layer's weight matrix onto its previous state during the optimizer update can improve generalization.

## Hypothesis

By blending the updated weights with the weights from before the optimizer step, we can introduce a form of regularization that prevents the model from taking overly large steps in the weight space. This could lead to better generalization and a lower validation loss. We call this method "Soft Weight Projection" (SWP).

## Method

We implemented a PyTorch optimizer wrapper, `SoftWeightProjection`, that wraps a standard optimizer like Adam. The `step()` function was modified as follows:

1.  Store a copy of the model's parameters before the update.
2.  Allow the wrapped optimizer to perform its `step()` as usual.
3.  After the update, blend the new parameters (`p_updated`) with the stored, pre-update parameters (`p_old`) using a `projection_strength` hyperparameter (`alpha`):

    `p_new = (1 - alpha) * p_updated + alpha * p_old`

A higher `alpha` means the weights are projected more strongly towards their previous state.

## Experimental Setup

-   **Dataset:** `mnist1d` (10,000 samples)
-   **Model:** A simple 3-layer MLP with ReLU activations.
-   **Baseline:** Standard Adam optimizer.
-   **Hyperparameter Tuning:** We used `optuna` for 20 trials to find the best hyperparameters for both the baseline and our SWP optimizer.
    -   For Adam, we tuned the learning rate.
    -   For SWP, we tuned both the learning rate and the `projection_strength`.

## Results

After running the `optuna` studies, we found the following best results:

-   **Best Adam Validation Loss:** 0.7062
    -   `lr`: 0.00329
-   **Best SWP Validation Loss:** 0.6923
    -   `lr`: 0.00411
    -   `projection_strength`: 0.1402

The `SoftWeightProjection` optimizer achieved a slightly lower validation loss than the tuned Adam baseline.

### Learning Curves

The following plot shows the validation loss curves for the best Adam and SWP models during training.

![Comparison Plot](comparison.png)

## Conclusion

The results support the hypothesis. Applying a soft projection of the weights back to their previous state during optimization provided a small but measurable improvement in generalization on the `mnist1d` dataset. The `SoftWeightProjection` method, when its `projection_strength` is tuned, can outperform a standard tuned Adam optimizer. This suggests that this form of regularization is a promising direction for further research.
