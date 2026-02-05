import subprocess
import optuna
import matplotlib.pyplot as plt
from train import get_data, train_model
from model import BaselineMLP, DSBMLP, FSBMLP
import torch
import os

def tune_and_train(model_name, model_class, data):
    print(f"Tuning {model_name}...")
    study = optuna.create_study(direction="maximize")
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        return train_model(model_class, lr, epochs=30, data=data)

    study.optimize(objective, n_trials=10)
    best_lr = study.best_params['lr']
    print(f"Best LR for {model_name}: {best_lr}")

    print(f"Final training for {model_name}...")
    # Run a few times with different seeds to get average and std
    accs = []
    for seed in range(3):
        torch.manual_seed(seed)
        save_path = f"dynamic_sparse_bottleneck_experiment/{model_name.replace(' ', '_')}_seed{seed}.pt"
        acc = train_model(model_class, best_lr, epochs=50, data=data, save_path=save_path)
        accs.append(acc)

    return accs, best_lr

if __name__ == "__main__":
    data = get_data()
    results = {}

    models = {
        "Baseline": BaselineMLP,
        "FSB": FSBMLP,
        "DSB": DSBMLP
    }

    for name, cls in models.items():
        accs, best_lr = tune_and_train(name, cls, data)
        results[name] = {"accs": accs, "best_lr": best_lr}

    print("\nResults Summary:")
    for name, res in results.items():
        avg = sum(res['accs']) / len(res['accs'])
        std = (sum((x - avg)**2 for x in res['accs']) / len(res['accs']))**0.5
        print(f"{name}: {avg:.2f}% +- {std:.2f}% (Best LR: {res['best_lr']:.5f})")

    # Plotting
    names = list(results.keys())
    avgs = [sum(results[n]['accs']) / len(results[n]['accs']) for n in names]
    stds = [(sum((x - avgs[i])**2 for x in results[n]['accs']) / len(results[n]['accs']))**0.5 for i, n in enumerate(names)]

    plt.figure(figsize=(10, 6))
    plt.bar(names, avgs, yerr=stds, capsize=5, color=['gray', 'blue', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of MLP variants on MNIST-1D')
    plt.ylim(min(avgs) - 5, max(avgs) + 5)
    plt.savefig('dynamic_sparse_bottleneck_experiment/results.png')
    plt.close()
