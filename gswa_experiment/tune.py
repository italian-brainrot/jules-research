import torch
import optuna
import mnist1d
from gswa_experiment.model import MLP
from gswa_experiment.trainer import train, evaluate
from light_dataloader import TensorDataLoader

def get_data():
    args = mnist1d.get_dataset_args()
    data = mnist1d.get_dataset(args, path='gswa_experiment/mnist1d_data.pkl')
    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Split train into train/val
    n_train = int(0.8 * len(x_train))
    train_ds = (x_train[:n_train], y_train[:n_train])
    val_ds = (x_train[n_train:], y_train[n_train:])

    return train_ds, val_ds, (x_test, y_test)

def objective(trial, mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, val_ds, test_ds = get_data()

    train_loader = TensorDataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = TensorDataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = TensorDataLoader(test_ds, batch_size=128, shuffle=False)

    model = MLP().to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Fast training for tuning
    best_model, history = train(model, train_loader, val_loader, optimizer, device, epochs=20, mode=mode)
    acc = evaluate(best_model, test_loader, device)
    return acc

if __name__ == "__main__":
    for mode in ['Adam', 'SWA', 'GSWA']:
        print(f"Tuning {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode), n_trials=15)
        print(f"Best {mode} LR: {study.best_params['lr']}, Best Acc: {study.best_value}")
