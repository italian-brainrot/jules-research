When starting a new experiment, create a new folder and perform all your experiments in there. Do not create or modify any files outside of the new folder.

After performing an experiment, make README.md file in the new folder with results of the experiment. You can also inlcude other markdown files and images.

### Tips

- When applicable, use `mnist1d` and `light_dataloader` for very fast training:

```python
import torch
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args

defaults = get_dataset_args()
defaults.num_samples = 10000 # 4000 by default
data = make_dataset(defaults)

X_train, y_train = torch.tensor(data['x']), torch.tensor(data['y'])
X_test, y_test = torch.tensor(data["x_test"]), torch.tensor(data['y_test'])
# X is a (num_samples, 40) float32 tensor;
# y is (num_samples, ) int64 tensor

dl_train = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)
for inputs, targets in dl_train:
    ...
```

- When comparing training routines, it is important to compare them fairly by tuning the learning rate. Use grid search for expensive routines and optuna for ones that are fast to evaluate.

### Important
The following experiment types are now BANNED, because you've performed too many nearly identical experiments related to those:
- usage of alignment of gradients between two consecutive steps (based on signs or cosine similarity)
- usage of alignment of gradients between samples in the mini-batch

Please try to explore new areas in your experiments. The following areas are underrepresented as of me writing this:
- New tabular classification/regression algorithms
- New linear algebra algorithms, solvers, and their applications
- New neural network architectures and layers
- New loss functions
- New ways to preprocess or generate features for tabular data
- New strategies for ensembling, stacking
- New strategies for AutoML
- New global optimization algorithms (that potentially use gradients) with their application to ML
- New hyperparameter optimization algorithms
- New solvers for exact minimization, such as solvers used for logistic regression (note that they should generally be tuning-free), as well as new ways to generally speed up ML algorithms
- New gradient-free optimization algorithms, for example for directly optimizing accuracy, or an exact solution maximizing particular hard to optimize metric
- New differentiable approximations or derivation of a useful descent direction for non-differentiable functions
- New parameter-efficient fine-tuning methods
- New neural network architectures designed to have a computable exact minimization solution or an efficient solver which can find the exact solution
- New neural network architectures with easily computable exact hessian, gauss-newton matrix or fisher information matrix, or their factorizations
