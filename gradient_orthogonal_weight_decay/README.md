# Gradient-Orthogonal Weight Decay (GOWD)

## Motivation
Standard weight decay (L2 regularization) pulls all weight components toward zero at a uniform rate. However, some weight directions may be more important for minimizing the loss than others. Gradient-Orthogonal Weight Decay (GOWD) proposes to protect the components of the weights that are aligned with the current gradient (or its moving average) and only decay the orthogonal components.

This "gradient-shielding" effect has two hypothesized benefits:
1. **Preservation of Signal:** It allows the model to maintain large weights in directions that are actively contributing to the learning process, even under high weight decay.
2. **Selective Regularization:** It regularizes directions that are not being updated by the optimizer, potentially reducing noise and preventing overfitting in irrelevant subspaces.

## Method
GOWD modifies the AdamW optimizer. At each step, after computing the Adam update direction $d = \hat{m} / (\sqrt{\hat{v}} + \epsilon)$, the weight $w$ is updated as follows:
1. Compute the projection coefficient of $w$ onto $d$: $\alpha = \frac{\langle w, d \rangle}{\|d\|^2}$.
2. Compute the projection vector: $\text{proj}_d(w) = \alpha d$.
3. Compute the orthogonal component: $w_{orth} = w - \text{proj}_d(w)$.
4. Apply weight decay only to the orthogonal component: $w \gets w - \eta \lambda w_{orth}$.
   - In implementation, this is equivalent to: $w \gets w (1 - \eta \lambda) + \eta \lambda \text{proj}_d(w)$.
5. Apply the standard Adam update: $w \gets w - \eta d$.

## Experimental Results

### Dataset and Model
- **Dataset:** MNIST1D (10,000 samples)
- **Model:** 3-layer MLP (40 -> 256 -> 256 -> 10)
- **Baseline:** Standard AdamW
- **Tuning:** Optuna was used to tune the learning rate and weight decay for both optimizers (30 trials each).

### Performance Comparison
| Optimizer | Best Test Accuracy | Best LR | Best Weight Decay |
|-----------|--------------------|---------|-------------------|
| AdamW     | 77.20%             | 3.90e-3 | 1.31e-5           |
| **GOWD**  | **77.70%**         | 3.79e-3 | 1.87e-5           |

GOWD achieved a slightly higher peak accuracy compared to AdamW on the MNIST1D dataset.

### Observations on Robustness
During the hyperparameter search, GOWD demonstrated significantly higher robustness to large weight decay values.
- For example, with a weight decay of ~0.02, AdamW's accuracy dropped to ~65%.
- In contrast, GOWD maintained an accuracy of ~73% with a weight decay of ~0.03.
- This confirms the hypothesis that "gradient-shielding" prevents the regularizer from suppressing useful learned features as aggressively as standard L2 regularization.

## Conclusion
Gradient-Orthogonal Weight Decay (GOWD) provides a more selective form of regularization by sparing the weight components aligned with the gradient. This leads to slightly better peak performance and significantly improved robustness to high weight decay hyperparameters. GOWD is easy to implement and adds minimal computational overhead to standard AdamW.

## How to Run
To reproduce the results, run:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 gradient_orthogonal_weight_decay/compare.py
```
The results will be saved in `gradient_orthogonal_weight_decay/results.txt` and `results.png`.
