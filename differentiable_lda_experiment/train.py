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
from sklearn.decomposition import PCA
from differentiable_lda_experiment.model import LDAClassifier
from differentiable_lda_experiment.lda import DLDALoss

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

    dl_train = TensorDataLoader((X_train[train_idx], y_train[train_idx]), batch_size=256, shuffle=True)
    dl_val = TensorDataLoader((X_train[val_idx], y_train[val_idx]), batch_size=256, shuffle=False)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

    return dl_train, dl_val, dl_test

def train_epoch(model, dl, optimizer, lda_loss_fn=None, lambda_lda=0.0, device='cpu'):
    model.train()
    total_loss = 0
    total_ce = 0
    total_lda = 0
    correct = 0
    total = 0

    for inputs, targets in dl:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if lambda_lda > 0 and lda_loss_fn is not None:
            logits, feat = model(inputs, return_features=True)
            ce_loss = F.cross_entropy(logits, targets)
            lda_loss = lda_loss_fn(feat, targets)
            loss = ce_loss + lambda_lda * lda_loss
        else:
            logits = model(inputs)
            ce_loss = F.cross_entropy(logits, targets)
            lda_loss = torch.tensor(0.0, device=device)
            loss = ce_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_ce += ce_loss.item() * inputs.size(0)
        total_lda += lda_loss.item() * inputs.size(0)

        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, total_ce / total, total_lda / total, correct / total

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
    dl_train, dl_val, _ = get_data()
    model = LDAClassifier(40, 128, 10).to(device)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    lambda_lda = 0.0
    lda_loss_fn = None
    if mode == "lda":
        lambda_lda = trial.suggest_float("lambda_lda", 1e-4, 1.0, log=True)
        lda_loss_fn = DLDALoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    for epoch in range(20):
        train_epoch(model, dl_train, optimizer, lda_loss_fn, lambda_lda, device)
        val_acc = validate(model, dl_val, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "eval"], default="tune")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.mode == "tune":
        params = {}
        for mode in ["baseline", "lda"]:
            print(f"Tuning {mode}...")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, mode, device), n_trials=args.trials)
            params[mode] = study.best_params
            print(f"Best params for {mode}: {study.best_params}")

        with open("differentiable_lda_experiment/best_params.json", "w") as f:
            json.dump(params, f)

    elif args.mode == "eval":
        if not os.path.exists("differentiable_lda_experiment/best_params.json"):
            print("Please run tuning first.")
            return

        with open("differentiable_lda_experiment/best_params.json", "r") as f:
            params = json.load(f)

        results = {}
        dl_train, dl_val, dl_test = get_data()

        for mode in ["baseline", "lda"]:
            print(f"Evaluating {mode}...")
            test_accs = []
            histories = []

            for seed in range(5):
                torch.manual_seed(seed)
                model = LDAClassifier(40, 128, 10).to(device)
                mode_params = params[mode]
                optimizer = optim.AdamW(model.parameters(), lr=mode_params["lr"], weight_decay=mode_params["weight_decay"])

                lambda_lda = mode_params.get("lambda_lda", 0.0)
                lda_loss_fn = DLDALoss() if lambda_lda > 0 else None

                history = {"train_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
                best_val_acc = 0
                final_test_acc = 0

                for epoch in range(args.epochs):
                    train_loss, train_ce, train_lda, train_acc = train_epoch(model, dl_train, optimizer, lda_loss_fn, lambda_lda, device)
                    val_acc = validate(model, dl_val, device)
                    test_acc = validate(model, dl_test, device)

                    history["train_loss"].append(train_loss)
                    history["train_acc"].append(train_acc)
                    history["val_acc"].append(val_acc)
                    history["test_acc"].append(test_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc

                test_accs.append(final_test_acc)
                histories.append(history)
                print(f"Seed {seed} complete. Final Test Acc: {final_test_acc:.4f}")

            results[mode] = {
                "mean_test_acc": np.mean(test_accs),
                "std_test_acc": np.std(test_accs),
                "histories": histories
            }
            print(f"Mode {mode}: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")

        # Visualization of features
        plt.figure(figsize=(15, 6))
        for i, mode in enumerate(["baseline", "lda"]):
            # Reload best model for first seed to visualize
            torch.manual_seed(0)
            model = LDAClassifier(40, 128, 10).to(device)
            mode_params = params[mode]
            optimizer = optim.AdamW(model.parameters(), lr=mode_params["lr"], weight_decay=mode_params["weight_decay"])
            lambda_lda = mode_params.get("lambda_lda", 0.0)
            lda_loss_fn = DLDALoss() if lambda_lda > 0 else None

            # Re-train for a few epochs for visualization if not already at best
            for epoch in range(args.epochs):
                train_epoch(model, dl_train, optimizer, lda_loss_fn, lambda_lda, device)

            model.eval()
            all_feats = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in dl_test:
                    _, feats = model(inputs.to(device), return_features=True)
                    all_feats.append(feats.cpu().numpy())
                    all_targets.append(targets.numpy())

            all_feats = np.concatenate(all_feats, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            pca = PCA(n_components=2)
            feats_2d = pca.fit_transform(all_feats)

            plt.subplot(1, 2, i+1)
            scatter = plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=all_targets, cmap='tab10', alpha=0.5)
            plt.title(f"PCA of {mode} Features")
            plt.colorbar(scatter)

        plt.tight_layout()
        plt.savefig("differentiable_lda_experiment/results.png")

        # Save results to txt
        with open("differentiable_lda_experiment/results.txt", "w") as f:
            f.write("Differentiable LDA Regularization Results\n")
            f.write("=========================================\n")
            for mode in ["baseline", "lda"]:
                f.write(f"{mode}: {results[mode]['mean_test_acc']*100:.2f}% +/- {results[mode]['std_test_acc']*100:.2f}%\n")
                f.write(f"Best Params: {params[mode]}\n\n")

if __name__ == "__main__":
    main()
