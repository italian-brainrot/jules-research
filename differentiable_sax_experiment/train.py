import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import SAXAugmentedMLP, SAXNet, BaselineMLP
import os

def train_model(model, dl_train, dl_test, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in dl_train:
            optimizer.zero_grad()
            out = model(x.float())
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_test:
                out = model(x.float())
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
    return best_acc

def get_data(num_samples=10000):
    defaults = get_dataset_args()
    defaults.num_samples = num_samples
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]), torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)
    return dl_train, dl_test

def objective_baseline(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    dl_train, dl_test = get_data()
    model = BaselineMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
    return train_model(model, dl_train, dl_test, epochs=30, lr=lr)

def objective_sax_net(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    num_segments = trial.suggest_int("num_segments", 4, 20)
    alphabet_size = trial.suggest_int("alphabet_size", 2, 8)
    dl_train, dl_test = get_data()
    model = SAXNet(input_dim=40, hidden_dim=hidden_dim, output_dim=10,
                  num_segments=num_segments, alphabet_size=alphabet_size)
    return train_model(model, dl_train, dl_test, epochs=30, lr=lr)

def objective_sax_augmented(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    num_segments = trial.suggest_int("num_segments", 4, 20)
    alphabet_size = trial.suggest_int("alphabet_size", 2, 8)
    dl_train, dl_test = get_data()
    model = SAXAugmentedMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10,
                           num_segments=num_segments, alphabet_size=alphabet_size)
    return train_model(model, dl_train, dl_test, epochs=30, lr=lr)

if __name__ == "__main__":
    n_trials = 10

    print("Tuning Baseline MLP...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(objective_baseline, n_trials=n_trials)

    print("Tuning SAX Net...")
    study_sax_net = optuna.create_study(direction="maximize")
    study_sax_net.optimize(objective_sax_net, n_trials=n_trials)

    print("Tuning SAX Augmented MLP...")
    study_sax_aug = optuna.create_study(direction="maximize")
    study_sax_aug.optimize(objective_sax_augmented, n_trials=n_trials)

    results = {
        "Baseline": study_baseline.best_value,
        "SAX Net": study_sax_net.best_value,
        "SAX Augmented": study_sax_aug.best_value
    }

    print("\nResults:")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

    # Final Evaluation with multiple seeds for the best models
    final_results = {}
    seeds = [42, 43, 44]
    studies = {
        "Baseline": study_baseline,
        "SAX Net": study_sax_net,
        "SAX Augmented": study_sax_aug
    }

    for name, study, model_cls in [
        ("Baseline", study_baseline, BaselineMLP),
        ("SAX Net", study_sax_net, SAXNet),
        ("SAX Augmented", study_sax_aug, SAXAugmentedMLP)
    ]:
        accs = []
        for seed in seeds:
            torch.manual_seed(seed)
            dl_train, dl_test = get_data()
            params = study.best_params.copy()
            lr = params.pop("lr")
            model = model_cls(input_dim=40, output_dim=10, **params)
            acc = train_model(model, dl_train, dl_test, epochs=50, lr=lr)
            accs.append(acc)
        final_results[name] = accs

    # Save results
    with open("differentiable_sax_experiment/results.txt", "w") as f:
        for name, accs in final_results.items():
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            f.write(f"{name}: {mean_acc:.4f} +/- {std_acc:.4f}\n")
            f.write(f"Best params: {studies[name].best_params}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    names = list(final_results.keys())
    means = [np.mean(final_results[n]) for n in names]
    stds = [np.std(final_results[n]) for n in names]
    plt.bar(names, means, yerr=stds, capsize=10, color=['gray', 'blue', 'green'])
    plt.ylabel("Accuracy")
    plt.title("Comparison of Models on MNIST-1D")
    plt.savefig("differentiable_sax_experiment/results.png")
