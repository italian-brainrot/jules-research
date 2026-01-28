# Trajectory Coherence Regularization Experiment

## Hypothesis

Penalizing sharp reversals in the weight update trajectory will discourage parameter oscillation, leading to smoother convergence and improved generalization. The regularization term is based on the cosine similarity between the weight update vector of the current step and the previous step. A penalty is applied when the similarity is negative, indicating a reversal in direction.

## Methodology

A simple MLP model was trained on the `mnist1d` dataset. Two training routines were compared:
1.  **Baseline:** A standard Adam optimizer.
2.  **Regularized:** Adam optimizer with the addition of the Trajectory Coherence Regularization term.

Optuna was used to perform a fair comparison by tuning the hyperparameters for each method. For the baseline, the learning rate was tuned. For the regularized version, both the learning rate and the regularization strength (`lambda`) were tuned. Each study was run for 10 trials.

## Results

The following best results were obtained from the experiment:

*   **Best baseline validation loss:** `1.3566`
    *   `lr`: `0.0147`
*   **Best regularized validation loss:** `1.4007`
    *   `lr`: `0.0710`
    *   `lam`: `0.0702`

## Conclusion

Even with the corrected implementation, the Trajectory Coherence Regularization did not improve the performance of the Adam optimizer on the `mnist1d` dataset. The baseline model still achieved a lower validation loss. The hypothesis that this form of regularization would lead to better generalization is not supported by these results. This suggests that for this particular problem, smoothing the weight trajectory by penalizing direction changes does not provide a benefit and may even slightly hinder the optimization process.
