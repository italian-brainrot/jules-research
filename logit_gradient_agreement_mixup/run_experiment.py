import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from logit_gradient_agreement_mixup.model import MLP
from logit_gradient_agreement_mixup.data import get_data, get_loaders
from logit_gradient_agreement_mixup.train_utils import mixup_data, lgam_mixup_data, mixup_criterion
from logit_gradient_agreement_mixup.tune import train_epoch, evaluate, objective

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_final_experiment():
    data = get_data()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    train_loader, val_loader, test_loader = get_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128)

    modes = ['Baseline', 'Mixup', 'LGAM']
    best_params = {}

    # 1. Tuning
    for mode in modes:
        print(f"Tuning {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data), n_trials=20)
        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {study.best_params}")

    # 2. Final Training with multiple seeds
    seeds = [42, 43, 44]
    results = {mode: [] for mode in modes}
    histories = {mode: [] for mode in modes}

    for mode in modes:
        print(f"Final training for {mode}...")
        params = best_params[mode]
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = MLP().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            criterion = nn.CrossEntropyLoss()

            history = {'train_loss': [], 'val_acc': [], 'test_acc': []}
            best_val_acc = 0
            test_acc_at_best_val = 0

            epochs = 50
            for epoch in range(epochs):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, mode,
                                         params.get('alpha', 1.0), params.get('gamma', 1.0))
                val_acc = evaluate(model, val_loader)
                test_acc = evaluate(model, test_loader)

                history['train_loss'].append(train_loss)
                history['val_acc'].append(val_acc)
                history['test_acc'].append(test_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc_at_best_val = test_acc

            results[mode].append(test_acc_at_best_val)
            histories[mode].append(history)
            print(f"Seed {seed}, Best Val Acc: {best_val_acc:.4f}, Test Acc at Best Val: {test_acc_at_best_val:.4f}")

    # 3. Summarize and Plot
    print("\nFinal Results:")
    for mode in modes:
        mean_acc = np.mean(results[mode])
        std_acc = np.std(results[mode])
        print(f"{mode}: {mean_acc:.4f} ± {std_acc:.4f}")

    # Plotting
    plt.figure(figsize=(15, 5))

    # Train Loss
    plt.subplot(1, 2, 1)
    for mode in modes:
        avg_loss = np.mean([h['train_loss'] for h in histories[mode]], axis=0)
        plt.plot(avg_loss, label=mode)
    plt.title('Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Test Accuracy
    plt.subplot(1, 2, 2)
    for mode in modes:
        avg_test_acc = np.mean([h['test_acc'] for h in histories[mode]], axis=0)
        plt.plot(avg_test_acc, label=mode)
    plt.title('Average Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('logit_gradient_agreement_mixup/comparison.png')

    # Save results to text
    with open('logit_gradient_agreement_mixup/results.txt', 'w') as f:
        f.write("Final Results:\n")
        for mode in modes:
            mean_acc = np.mean(results[mode])
            std_acc = np.std(results[mode])
            f.write(f"{mode}: {mean_acc:.4f} ± {std_acc:.4f}\n")
            f.write(f"Best Params: {best_params[mode]}\n")
            f.write(f"Seed Results: {results[mode]}\n\n")

if __name__ == "__main__":
    run_final_experiment()
