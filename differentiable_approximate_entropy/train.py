import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import BaselineMLP, DSampEnAugmentedMLP
import optuna
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, X_test, y_test, lr, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_acc = (test_outputs.argmax(1) == y_test).float().mean().item()

        history.append(test_acc)
    return history

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    if model_type == "baseline":
        model = BaselineMLP(input_dim=40, hidden_dim=256, output_dim=10)
    else:
        model = DSampEnAugmentedMLP(input_dim=40, hidden_dim=256, output_dim=10)

    history = train_model(model, dl_train, X_test, y_test, lr, epochs=15)
    return max(history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "eval"], default="eval")
    parser.add_argument("--model", choices=["baseline", "dsampen"], default="baseline")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = get_data()

    if args.mode == "tune":
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, args.model, X_train, y_train, X_test, y_test), n_trials=10)
        print(f"Best trial: {study.best_trial.params}")

    elif args.mode == "eval":
        all_histories = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
            if args.model == "baseline":
                model = BaselineMLP(input_dim=40, hidden_dim=256, output_dim=10)
            else:
                model = DSampEnAugmentedMLP(input_dim=40, hidden_dim=256, output_dim=10)

            history = train_model(model, dl_train, X_test, y_test, args.lr, epochs=30)
            all_histories.append(history)
            print(f"Seed {seed} final acc: {history[-1]:.4f}")

        all_histories = np.array(all_histories)
        mean_acc = all_histories.mean(axis=0)
        std_acc = all_histories.std(axis=0)

        print(f"Final results for {args.model}: {mean_acc[-1]:.4f} +/- {std_acc[-1]:.4f}")

        # Save results to a file for later summary
        with open(f"differentiable_approximate_entropy/results_{args.model}.txt", "w") as f:
            f.write(f"Mean: {mean_acc[-1]}\n")
            f.write(f"Std: {std_acc[-1]}\n")
            f.write(f"LR: {args.lr}\n")
            f.write(f"History: {mean_acc.tolist()}\n")
