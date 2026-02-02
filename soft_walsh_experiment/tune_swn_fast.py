import optuna
import json
from train import get_data, train_model
from light_dataloader import TensorDataLoader
from model import SoftWalshNetwork
import torch

X_train, y_train, X_test, y_test = get_data()

def objective(trial):
    train_loader = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    nt = trial.suggest_int('nt', 64, 256)
    sl = trial.suggest_float('sl', 1e-6, 1e-2, log=True)
    dp = trial.suggest_categorical('dp', [True, False])
    iscl = trial.suggest_float('iscl', 1e-3, 0.5, log=True)

    model = SoftWalshNetwork(40, nt, 10, deep=dp, init_scale=iscl)
    return train_model(model, train_loader, test_loader, epochs=10, lr=lr, weight_decay=wd, sparsity_lambda=sl)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
with open('swn_params.json', 'w') as f:
    json.dump(study.best_params, f)
print(study.best_value)
