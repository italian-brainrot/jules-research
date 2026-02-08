import torch
import torch.nn as nn
import torch.optim as optim
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import optuna
import numpy as np
from model import INGOMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_one_run(model_type, lr, lambda_penalty, epochs=30, data=None, seed=42):
    torch.manual_seed(seed)
    X_train, y_train, X_test, y_test = data
    model = INGOMLP(input_dim=40, hidden_dims=[256, 256, 256], output_dim=10, penalty_type=model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for x, y in dl_train:
            optimizer.zero_grad()
            if model_type == 'none':
                logits = model(x)
                loss = criterion(logits, y)
            else:
                logits, penalty = model(x)
                loss = criterion(logits, y) + lambda_penalty * penalty
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        if model_type != 'none': test_logits = test_outputs[0]
        else: test_logits = test_outputs
        preds = torch.argmax(test_logits, dim=1)
        acc = (preds == y_test).float().mean().item() * 100
    return acc

def tune_model(model_type, data, n_trials=5):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lambda_penalty = 0.0
        if model_type != 'none':
            lambda_penalty = trial.suggest_float("lambda", 1e-5, 0.1, log=True)
        return train_one_run(model_type, lr, lambda_penalty, epochs=20, data=data, seed=42)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

if __name__ == "__main__":
    data = get_data()
    model_types = ['none', 'ingo']
    best_hparams = {}
    for m_type in model_types:
        print(f"Tuning {m_type}...")
        best_hparams[m_type] = tune_model(m_type, data, n_trials=5)
        print(f"Best hparams for {m_type}: {best_hparams[m_type]}")
    final_results = {}
    for m_type in model_types:
        print(f"Final evaluation for {m_type}...")
        accs = []
        lr = best_hparams[m_type]['lr']
        lambda_penalty = best_hparams[m_type].get('lambda', 0.0)
        for seed in range(3):
            acc = train_one_run(m_type, lr, lambda_penalty, epochs=40, data=data, seed=seed)
            accs.append(acc)
        final_results[m_type] = accs
        print(f"{m_type}: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
    with open('ingo_experiment/results_final.txt', 'w') as f:
        for m_type, accs in final_results.items():
            f.write(f"{m_type}: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}% (Hparams: {best_hparams[m_type]})\n")
