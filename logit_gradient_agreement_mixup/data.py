import torch
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader

def get_data(num_samples=10000):
    args = get_dataset_args()
    args.num_samples = num_samples
    data = make_dataset(args)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    # Validation split
    n_train = int(0.9 * len(X_train))
    X_val = X_train[n_train:]
    y_val = y_train[n_train:]
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128):
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
