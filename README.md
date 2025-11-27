When starting a new experiment, create a new folder and perform all your experiments in there. Do not create or modify any files outside of the new folder.

After performing an experiment, make README.md file in the new folder with results of the experiment. You can also inlcude other markdown files and images.

### Tips

Use `mnist1d` and `light_dataloader` for very fast training:

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

