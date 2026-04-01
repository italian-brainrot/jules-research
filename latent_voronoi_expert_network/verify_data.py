import torch
from mnist1d.data import make_dataset, get_dataset_args

def verify_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)
    X_train = torch.tensor(data['x'])
    y_train = torch.tensor(data['y'])
    X_test = torch.tensor(data['x_test'])
    y_test = torch.tensor(data['y_test'])
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

if __name__ == "__main__":
    verify_data()
