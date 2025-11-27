Please ignore all folders from previous experiments in this repo, create a new folder and perform all your experiments in there. Do not create or modify any files outside of the new folder.

After performing an experiment, make README.md file in the new folder with results of the experiment. You can also inlcude other markdown files and images.

### Tips

Use `mnist1d` and `light_dataloader` for very fast training:

```python
import torch
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args

defaults = get_dataset_args()
data = make_dataset(defaults)
X, y = torch.tensor(data['x']), torch.tensor(data['y'])
# X is a (4000, 40) float32 tensor;
# y is (4000, ) int64 tensor

dl = TensorDataLoader((X, y), batch_size=32, shuffle=True)
for inputs, targets in dl:
    ...
```

