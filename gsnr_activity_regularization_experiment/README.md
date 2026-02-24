# GSNR-based Activity Regularization (GWAR) Experiment

## Hypothesis
Penalizing neurons that receive inconsistent gradient signals across a batch encourages the network to rely on more robust features. The GSNR of the activation gradients is used to weight the activity regularization penalty: $L_{GWAR} = \lambda \sum_{layer} \sum_{i} (1 - GSNR_i) \cdot \text{mean}_b(a_{b,i}^2)$

## Results
| Mode | Test Accuracy | Best Hyperparameters |
| --- | --- | --- |
| Baseline | 0.7079 ± 0.0061 | {'lr': 0.006023669064640673, 'weight_decay': 5.817554511089864e-06} |
| GWAR | 0.7560 ± 0.0138 | {'lr': 0.0059155753432051, 'weight_decay': 0.01151550510647973, 'lambda_gwar': 0.00887933419699952} |

## Discussion
GWAR outperformed the baseline, suggesting that gradient-consistency-based activity regularization is beneficial.
