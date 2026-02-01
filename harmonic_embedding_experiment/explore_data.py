import torch
from mnist1d.data import make_dataset, get_dataset_args

defaults = get_dataset_args()
defaults.num_samples = 1000
data = make_dataset(defaults)

X = torch.tensor(data['x'])
print(f"X shape: {X.shape}")
print(f"X min: {X.min()}, max: {X.max()}")
print(f"X mean: {X.mean()}, std: {X.std()}")
