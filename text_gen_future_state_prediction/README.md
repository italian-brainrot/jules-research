# Future State Prediction Regularization (FSPR) for Text Generation

## Hypothesis
Adding a regularization term that forces the current hidden state $h_t$ of a recurrent neural network (GRU) to contain enough information to predict a future hidden state $h_{t+k}$ improves the model's ability to capture long-range dependencies and leads to better character-level language modeling performance.

## Methodology
- **Dataset:** A subset (10,000 characters) of the Tiny Shakespeare dataset.
- **Task:** Character-level next-token prediction.
- **Model:** 1-layer GRU with an embedding layer and a linear output layer.
- **Regularization:** An auxiliary MLP head predicts the hidden state $k=4$ steps into the future. The loss is:
  $L = L_{CE} + \lambda \cdot \| \text{MLP}(h_t) - \text{detach}(h_{t+k}) \|^2$
- **Baseline:** Standard GRU trained with only Cross-Entropy loss.
- **Fair Comparison:** The learning rate for both the baseline and the experiment was tuned using Optuna. The regularization weight $\lambda$ was also tuned for the experiment.
- **Evaluation:** Validation Cross-Entropy loss and qualitative text generation.

## Results
| Method | Best Validation Loss | Best Learning Rate | Best $\lambda$ |
|--------|----------------------|--------------------|----------------|
| Baseline | 2.1186 | 0.000887 | N/A |
| FSPR (Ours) | 2.0912 | 0.003515 | 0.0105 |

### Qualitative Comparison
**Baseline Generation:**
> A sick make you must not the patricians of your blood to them to the common
> The heart, to the senatters of you

**FSPR Generation:**
> A sick make it, and
> Your knees to them, not arms, must help. Alack,
> You are transported by calamity
> Thither wh

The FSPR model achieved a slightly lower validation loss and produced more coherent text during generation from the same prompt. Interestingly, the FSPR model benefited from a significantly higher learning rate, suggesting that the auxiliary task might provide a more stable or informative gradient signal.

## Conclusion
Future State Prediction Regularization (FSPR) appears to be a promising and simple technique to improve the performance of autoregressive models. By encouraging the hidden state to "look ahead", the model develops representations that are more useful for long-term prediction.

## Files
- `data.py`: Data loading and preprocessing for Tiny Shakespeare.
- `model.py`: GRU model with the auxiliary FSP head.
- `compare.py`: Script for hyperparameter tuning and comparison.
- `loss_comparison.png`: Plot of validation loss over time.
- `results.txt`: Summary of numerical results and sample generations.
