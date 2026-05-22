# Differentiable Cepstrum Experiment

This experiment investigates the utility of **differentiable real cepstrum** features for 1D signal classification using the MNIST-1D dataset.

## Hypothesis
The real cepstrum is defined as the inverse Fourier transform of the log-magnitude spectrum of a signal:
$$C(x) = \text{IDFT}(\ln(|\text{DFT}(x)| + \epsilon))$$
It is widely used in speech processing and seismology to separate the source (excitation) from the filter (system response). We hypothesize that these features can provide complementary information to the raw signal, improving classification performance when used as input or augmentation for a neural network.

## Methodology
- **Dataset**: MNIST-1D (10,000 samples).
- **Architecture**: A 2-hidden-layer MLP (Hidden size 128) with ReLU activations.
- **Differentiable Layer**: Implemented a custom PyTorch layer using a manual DFT matrix for numerical stability and compatibility across different CPU/MKL environments.
- **Models Compared**:
    1. **BaselineMLP**: Acts directly on the raw signal (40 features).
    2. **CepstrumMLP**: Acts only on the real cepstrum (40 features).
    3. **CepstrumAugmentedMLP**: Acts on the concatenation of the raw signal and the real cepstrum (80 features).
- **Hyperparameter Tuning**: Used Optuna to tune the learning rate for each model independently over 5 trials.
- **Training**: Each model was trained for 30 epochs using its best learning rate with the Adam optimizer.

## Results

| Model | Best Learning Rate | Final Test Accuracy |
|-------|--------------------|---------------------|
| Baseline | 0.001277 | 69.55% |
| Cepstrum | 0.002174 | 40.75% |
| CepstrumAugmented | 0.008873 | 72.55% |

The convergence plots (Training Loss and Test Accuracy) are available in `comparison.png`.

## Analysis
- **Augmentation Benefit**: The `CepstrumAugmentedMLP` outperformed the `BaselineMLP` by approximately 3%, suggesting that cepstral features do provide useful, non-redundant information that aids classification.
- **Standalone Cepstrum**: The `CepstrumMLP` achieved 40.75% accuracy. While significantly better than random guessing (10%), it is much lower than the baseline, indicating that the cepstrum alone discards too much information (like phase and exact temporal positioning) for this specific task.
- **Stability**: Implementing the FFT via a matrix multiplication was necessary to bypass intermittent `MKL FFT error` issues in the environment, ensuring the differentiability and reliability of the layer.

## Conclusion
The differentiable real cepstrum is an effective feature for 1D signal classification when used as an augmentation. While not a replacement for raw signal features in the MNIST-1D task, it provides a measurable performance boost, confirming its value in the deep learning toolbox for time-series and signal processing.
