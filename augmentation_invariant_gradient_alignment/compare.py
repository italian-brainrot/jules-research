import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from augmentation_invariant_gradient_alignment.train import MLP, get_data, train_epoch, evaluate, set_seed
from light_dataloader import TensorDataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_comparison(seeds=[42]):
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    results = {}

    for mode in ['Baseline', 'AIGA']:
        with open(f'augmentation_invariant_gradient_alignment/best_params_{mode}.json', 'r') as f:
            params = json.load(f)

        print(f"Running final evaluation for {mode}...")
        mode_test_accs = []
        mode_histories = []

        for seed in seeds:
            set_seed(seed)
            model = MLP().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            lambda_aiga = params.get('lambda_aiga', 0.0)

            history = {'train_loss': [], 'val_acc': [], 'test_acc': []}
            best_val_acc = 0
            test_acc_at_best_val = 0

            for epoch in range(25):
                train_loss = train_epoch(model, train_loader, optimizer, lambda_aiga, use_aug=True)
                val_acc = evaluate(model, val_loader)
                test_acc = evaluate(model, test_loader)

                history['train_loss'].append(train_loss)
                history['val_acc'].append(val_acc)
                history['test_acc'].append(test_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc_at_best_val = test_acc

            mode_test_accs.append(test_acc_at_best_val)
            mode_histories.append(history)
            print(f"Mode {mode}, Seed {seed}, Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc_at_best_val:.4f}")

        results[mode] = {
            'test_accs': mode_test_accs,
            'mean': np.mean(mode_test_accs),
            'std': np.std(mode_test_accs),
            'histories': mode_histories
        }

    # Save summary results
    with open('augmentation_invariant_gradient_alignment/results.txt', 'w') as f:
        for mode in results:
            res = results[mode]
            f.write(f"{mode}: {res['mean']:.4f} +/- {res['std']:.4f} (Accs: {res['test_accs']})\n")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for mode in results:
        avg_loss = np.mean([h['train_loss'] for h in results[mode]['histories']], axis=0)
        plt.plot(avg_loss, label=mode)
    plt.title('Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in results:
        avg_val_acc = np.mean([h['val_acc'] for h in results[mode]['histories']], axis=0)
        plt.plot(avg_val_acc, label=mode)
    plt.title('Average Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('augmentation_invariant_gradient_alignment/results.png')
    plt.close()

    print("\nResults Summary:")
    for mode in results:
        print(f"{mode}: {results[mode]['mean']:.4f} +/- {results[mode]['std']:.4f}")

if __name__ == '__main__':
    run_comparison()
