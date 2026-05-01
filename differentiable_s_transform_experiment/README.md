# Differentiable S-Transform Experiment

This experiment evaluates the effectiveness of the S-Transform (Stockwell Transform) as a differentiable feature extractor for signal classification on the MNIST-1D dataset.

## Method

The S-Transform provides a time-frequency representation that combines the advantages of the Short-Time Fourier Transform (STFT) and Wavelet Transform. It uses a frequency-dependent window width, which provides better resolution at higher frequencies.

In this experiment, we implemented a `DifferentiableSTLayer` that:
1. Computes the S-Transform using FFT-based frequency-domain convolution.
2. Supports a learnable `sigma` parameter that controls the window width.
3. Is fully differentiable with respect to both the input signal and the `sigma` parameter.

We compared four models:
- **Baseline**: A standard MLP with hidden dimension 256.
- **BaselineWide**: A standard MLP with hidden dimension 673 to match the parameter count of the `STAug` model (~488k parameters).
- **STAug**: An MLP that takes both the raw signal and the magnitude of its S-Transform as input.
- **ST**: An MLP that takes only the magnitude of the S-Transform as input.

## Results

The models were tuned using Optuna for 5 trials to find the optimal learning rate and then evaluated across 5 seeds.

| Model | Test Accuracy |
|-------|---------------|
| Baseline | 79.41% +/- 0.50% |
| BaselineWide | 81.67% +/- 0.35% |
| STAug | 90.09% +/- 2.34% |
| ST | 58.06% +/- 0.72% |

The `STAug` model significantly outperforms the baselines, achieving over 90% accuracy. This suggests that the S-Transform provides highly discriminative time-frequency features that are complementary to the raw time-domain signal. The standalone `ST` model performs poorly, likely because the phase information (which is discarded by taking the magnitude) or the original time-domain structure is important for this task.

## Files
- `model.py`: Implementation of the Differentiable S-Transform layer and the models.
- `test_logic.py`: Mathematical verification and gradient checks.
- `train.py`: Training script with hyperparameter tuning and multi-seed evaluation.
- `results.txt`: Raw results output.
- `train_output.log`: Execution log of the training process.
