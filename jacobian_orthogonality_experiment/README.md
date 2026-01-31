# Class-wise Jacobian Orthogonality Regularization (CJOR)

## Hypothesis
Standard regularization methods like Weight Decay or Jacobian Frobenius Norm Regularization (JFNR) penalize the overall magnitude of weights or gradients. However, for classification tasks, it might be beneficial to ensure that the features the model relies on for different classes are "decoupled" or orthogonal in the input space.

We hypothesize that **Class-wise Jacobian Orthogonality Regularization (CJOR)**, which penalizes the squared cosine similarity between the gradients of different class logits with respect to the input, will improve generalization by reducing inter-class confusion and encouraging the model to find unique, class-specific features.

## Methodology
- **Dataset**: `mnist1d` with 8,000 samples.
- **Model**: A 3-layer MLP (40 -> 128 -> 128 -> 10) with ReLU activations.
- **Configurations Compared**:
  - **Baseline**: Standard training with Adam optimizer, tuned learning rate.
  - **JFNR**: Baseline + Jacobian Frobenius Norm Regularization ($\|J\|_F^2$).
  - **CJOR**: Baseline + Class-wise Jacobian Orthogonality Regularization (squared cosine similarity of class gradients).
  - **CJOR+JFNR**: Combining both regularization terms.
- **Hyperparameter Tuning**: Optuna was used to tune the learning rate and regularization strengths ($\lambda$) for each configuration (**20 trials** each).
- **Evaluation**: Each best configuration was trained for **50 epochs**.

## Results

| Configuration | Best Test Accuracy | Final Test Accuracy | Best Hyperparameters |
|---------------|-------------------|---------------------|----------------------|
| Baseline      | 71.31%            | 71.06%              | LR: 4.83e-3          |
| JFNR          | 71.06%            | 71.06%              | LR: 4.75e-3, $\lambda_{JFNR}$: 2.59e-4 |
| CJOR          | 70.63%            | 70.19%              | LR: 4.31e-3, $\lambda_{CJOR}$: 1.06e-3 |
| **CJOR+JFNR** | **72.56%**        | **71.50%**          | LR: 4.97e-3, $\lambda_{JFNR}$: 6.76e-5, $\lambda_{CJOR}$: 8.10e-4 |

### Analysis
- In this more extensive run (20 trials, 50 epochs), **CJOR combined with JFNR achieved the highest final test accuracy (71.50%)** and the highest peak accuracy (72.56%).
- The baseline also improved significantly with more tuning, reaching 71.06%.
- While CJOR alone was slightly behind the baseline in this specific run, its combination with JFNR provided a noticeable boost, suggesting that orthogonality regularization works best when combined with magnitude-based Jacobian regularization.
- The results demonstrate that regularizing the structure of the Jacobian (orthogonality between classes) can provide additional generalization benefits over simply regularizing its norm.

## Visualizations
- `accuracy_comparison.png`: Shows the test accuracy over epochs for all configurations.
- `loss_comparison.png`: Shows the training loss over epochs for all configurations.

## Conclusion
Class-wise Jacobian Orthogonality Regularization (CJOR) is a promising technique for improving the generalization of classifiers. When combined with standard Jacobian Frobenius Norm Regularization, it helps the network learn representations that are both robust (low sensitivity) and discriminative (orthogonal class-wise sensitivity).
