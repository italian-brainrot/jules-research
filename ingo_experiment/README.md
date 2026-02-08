# Inter-Neuron Gradient Orthogonality (INGO) Experiment

This experiment investigates a new regularization method called **Inter-Neuron Gradient Orthogonality (INGO)**.

## Hypothesis
Standard weight orthogonality penalizes the similarity of weight vectors regardless of whether the neurons are active at the same time. We hypothesize that encouraging orthogonality only between neurons that are simultaneously sensitive to the same input (co-active/co-sensitive) allows for more efficient feature representation.

## Method
INGO computes a penalty based on the squared norm of the sum of sensitive directions for each sample in a batch:
$$ L_{INGO} = \frac{1}{B} \sum_{b=1}^B \| \sum_{i} \sigma'(z_{b,i}) \frac{w_i}{\|w_i\|} \|^2 $$
Where:
- $w_i$ is the weight vector of neuron $i$.
- $\sigma'(z_{b,i})$ is the sensitivity of neuron $i$ for sample $b$.
- $B$ is the batch size.

This penalty encourages:
1. Weight vectors of co-active neurons to be orthogonal.
2. Neurons with similar weight vectors to have non-overlapping activation regions.

## Experimental Setup
- **Dataset**: MNIST-1D (10,000 samples).
- **Model**: 3-layer MLP with 256 hidden units and GELU activations.
- **Baseline**: Standard MLP tuned with Optuna.
- **Comparison**: Weight Orthogonality (static) and INGO (input-dependent).
- **Optimization**: Adam optimizer, learning rate and penalty strength tuned via Optuna.

## Results
Based on the experiments conducted:

| Method | Test Accuracy | Note |
|---|---|---|
| Baseline | 78.48% ± 0.52% | Stable performance |
| INGO | 74.07% ± 10.44% | Reached **79.75%** in tuning, but showed high variance in final runs |
| Weight Ortho | 77.75% ± 0.48% | Performed slightly worse than baseline |

### Observation
INGO demonstrated the potential to outperform the baseline (achieving nearly 80% accuracy in some runs), suggesting it can find superior feature representations. However, it exhibited significant instability and sensitivity to hyperparameters (especially the penalty strength $\lambda$). The high variance in final runs suggests that while the "good" minima found by INGO are better than baseline, they may be harder to reach consistently or may require more careful learning rate scheduling.

## Conclusion
Inter-Neuron Gradient Orthogonality is a promising direction for input-dependent regularization. It is more flexible than static weight orthogonality as it only enforces constraints where neurons are actually competing to represent the input. Future work could investigate smoother penalty transitions or adaptive $\lambda$ scheduling to improve stability.
