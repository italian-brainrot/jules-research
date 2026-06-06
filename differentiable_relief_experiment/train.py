import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from differentiable_relief_experiment.model import get_model
from differentiable_relief_experiment.layer import DReliefLoss

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()

    # Val split
    idx = torch.randperm(len(X_train))
    train_size = int(0.8 * len(X_train))
    train_idx, val_idx = idx[:train_size], idx[train_size:]

    dl_train = TensorDataLoader((X_train[train_idx], y_train[train_idx]), batch_size=128, shuffle=True)
    dl_val = TensorDataLoader((X_train[val_idx], y_train[val_idx]), batch_size=128, shuffle=False)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    return dl_train, dl_val, dl_test

def train_epoch(model, dl, optimizer, relief_loss_fn=None, lambda_relief=0.0, device='cpu'):
    model.train()
    total_loss = 0
    total_ce = 0
    total_relief = 0
    correct = 0
    total = 0

    for inputs, targets in dl:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if lambda_relief > 0 and relief_loss_fn is not None:
            logits, feat = model(inputs, return_features=True)
            ce_loss = F.cross_entropy(logits, targets)
            relief_loss = relief_loss_fn(feat, targets)
            loss = ce_loss + lambda_relief * relief_loss
        else:
            logits = model(inputs)
            ce_loss = F.cross_entropy(logits, targets)
            relief_loss = torch.tensor(0.0, device=device)
            loss = ce_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_ce += ce_loss.item() * inputs.size(0)
        total_relief += relief_loss.item() * inputs.size(0)

        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, total_ce / total, total_relief / total, correct / total

def validate(model, dl, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dl:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

def objective(trial, mode, device):
    dl_train, dl_val, dl_test = get_data()
    model = get_model().to(device)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    lambda_relief = 0.0
    relief_loss_fn = None
    if mode == "relief":
        lambda_relief = trial.suggest_float("lambda_relief", 1e-4, 1.0, log=True)
        temp = trial.suggest_float("temperature", 1e-2, 10.0, log=True)
        relief_loss_fn = DReliefLoss(temperature=temp)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    for epoch in range(20):
        train_epoch(model, dl_train, optimizer, relief_loss_fn, lambda_relief, device)
        val_acc = validate(model, dl_val, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "eval"], default="tune")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.mode == "tune":
        params = {}
        for mode in ["baseline", "relief"]:
            print(f"Tuning {mode}...")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, mode, device), n_trials=args.trials)
            params[mode] = study.best_params
            print(f"Best params for {mode}: {study.best_params}")

        with open("differentiable_relief_experiment/best_params.json", "w") as f:
            json.dump(params, f)

    elif args.mode == "eval":
        if not os.path.exists("differentiable_relief_experiment/best_params.json"):
            print("Please run tuning first.")
            return

        with open("differentiable_relief_experiment/best_params.json", "r") as f:
            params = json.load(f)

        results = {}
        dl_train, dl_val, dl_test = get_data()

        for mode in ["baseline", "relief"]:
            print(f"Evaluating {mode}...")
            test_accs = []
            histories = []

            for seed in range(5):
                torch.manual_seed(seed)
                model = get_model().to(device)
                mode_params = params[mode]
                optimizer = optim.AdamW(model.parameters(), lr=mode_params["lr"], weight_decay=mode_params["weight_decay"])

                lambda_relief = mode_params.get("lambda_relief", 0.0)
                temp = mode_params.get("temperature", 1.0)
                relief_loss_fn = DReliefLoss(temperature=temp) if lambda_relief > 0 else None

                history = {"train_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
                best_val_acc = 0
                final_test_acc = 0

                for epoch in range(args.epochs):
                    train_loss, train_ce, train_relief, train_acc = train_epoch(model, dl_train, optimizer, relief_loss_fn, lambda_relief, device)
                    val_acc = validate(model, dl_val, device)
                    test_acc = validate(model, dl_test, device)

                    history["train_loss"].append(train_loss)
                    history["train_acc"].append(train_acc)
                    history["val_acc"].append(val_acc)
                    history["test_acc"].append(test_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc

                    if (epoch + 1) % 10 == 0:
                        print(f"Seed {seed}, Epoch {epoch+1}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

                test_accs.append(final_test_acc)
                histories.append(history)

            results[mode] = {
                "mean_test_acc": np.mean(test_accs),
                "std_test_acc": np.std(test_accs),
                "histories": histories
            }
            print(f"Mode {mode}: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")

        with open("differentiable_relief_experiment/results.json", "w") as f:
            json.dump(results, f)

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for mode in ["baseline", "relief"]:
            all_train_acc = np.array([h["train_acc"] for h in results[mode]["histories"]])
            plt.plot(all_train_acc.mean(axis=0), label=f"{mode} Train")
        plt.title("Training Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        for mode in ["baseline", "relief"]:
            all_test_acc = np.array([h["test_acc"] for h in results[mode]["histories"]])
            plt.plot(all_test_acc.mean(axis=0), label=f"{mode} Test")
        plt.title("Test Accuracy")
        plt.legend()
        plt.savefig("differentiable_relief_experiment/results.png")

if __name__ == "__main__":
    main()
