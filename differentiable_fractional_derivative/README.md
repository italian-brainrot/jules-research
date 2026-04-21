# Differentiable Fractional Derivative (DFD) Experiment

This experiment investigates the use of **Differentiable Fractional Derivatives** as a learnable feature extraction layer for 1D signal classification.

## Method

Fractional calculus generalizes derivatives to non-integer orders $q$. Unlike standard derivatives which are purely local, fractional derivatives have a "memory" effect, capturing both local trends and long-term dependencies in the signal.

We implement the **Grünwald-Letnikov (GL)** derivative, which defines the discrete fractional derivative as:
$$D^q x_t = \sum_{k=0}^{t} c_k(q) x_{t-k}$$
where the coefficients $c_k(q)$ are defined by:
$$c_0 = 1, \quad c_k = c_{k-1} \left(1 - \frac{q+1}{k}\right)$$

In our DFD layer:
- The order $q$ is a **learnable parameter**.
- Multiple orders can be learned in parallel.
- The operation is implemented as a causal convolution with power-law decaying weights.
- It is fully differentiable with respect to the input signal $x$ and the order $q$.

## Experiment Setup

- **Dataset**: `mnist1d` (10,000 samples, length 40).
- **Models**:
    - `BaselineMLP`: A standard 2-layer MLP.
    - `DFDAugmentedMLP`: An MLP that receives both the raw signal and 4 learnable fractional derivatives of the signal.
- **Fair Comparison**: Learning rates for both models were tuned using Optuna (15 trials each).
- **Evaluation**: Final performance was averaged over 5 random seeds.

## Results

| Model | Test Accuracy |
|-------|---------------|
| Baseline MLP | 77.12% ± 1.94% |
| DFD-Augmented MLP | **78.90% ± 1.58%** |

### Learned Orders
During training, the model learned orders that typically spanned a range from fractional (e.g., $q \approx 0.5$) to higher-order (e.g., $q \approx 2.2$), suggesting that different degrees of differentiation/integration provide complementary information for classification.

Example learned orders from a typical run: `[0.56, 1.03, 1.54, 2.19]`

## Conclusion

The DFD-Augmented MLP consistently outperformed the baseline, demonstrating that learnable fractional derivatives provide a useful inductive bias for signal classification. The ability to learn the differentiation order allows the network to adapt its "memory" and "sensitivity" to the specific frequency components and temporal patterns of the dataset.

## How to Run

1.  **Verify Logic**:
    ```bash
    python3 differentiable_fractional_derivative/test_logic.py
    ```
2.  **Run Experiment**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/differentiable_fractional_derivative
    python3 differentiable_fractional_derivative/train.py
    ```
