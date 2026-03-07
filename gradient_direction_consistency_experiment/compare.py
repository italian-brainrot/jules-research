import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import os
import ast
from gradient_direction_consistency_experiment.train import MLP, train_epoch, evaluate, get_data, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_final_training(mode, params, data, seeds=[42, 43, 44]):
    x_train, y_train, x_val, y_val, x_test, y_test = data
    train_loader = TensorDataLoader((x_train, y_train), batch_size=64, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=64, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=64, shuffle=False)

    lr = params['lr']
    wd = params['weight_decay']
    lambda_gdcr = params.get('lambda_gdcr', 0.0)

    all_seed_results = []

    for seed in seeds:
        set_seed(seed)
        model = MLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        best_val_acc = 0
        test_acc_at_best_val = 0

        history = {'train_loss': [], 'val_acc': [], 'test_acc': []}

        for epoch in range(50):
            loss = train_epoch(model, train_loader, optimizer, mode, lambda_gdcr, device)
            val_acc = evaluate(model, val_loader, device)
            test_acc = evaluate(model, test_loader, device)

            history['train_loss'].append(loss)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc

        all_seed_results.append({
            'history': history,
            'best_test_acc': test_acc_at_best_val
        })
        print(f"Mode: {mode}, Seed: {seed}, Best Test Acc: {test_acc_at_best_val:.4f}")

    return all_seed_results

if __name__ == "__main__":
    data = get_data()

    results = {}
    for mode in ['Baseline', 'GDCR']:
        with open(f'gradient_direction_consistency_experiment/best_params_{mode}.txt', 'r') as f:
            params = ast.literal_eval(f.read())

        print(f"Running final training for {mode}...")
        results[mode] = run_final_training(mode, params, data)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Training Loss Plot
    plt.subplot(1, 2, 1)
    for mode in ['Baseline', 'GDCR']:
        # Average history across seeds
        avg_loss = np.mean([res['history']['train_loss'] for res in results[mode]], axis=0)
        plt.plot(avg_loss, label=mode)
    plt.title('Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Test Accuracy Plot
    plt.subplot(1, 2, 2)
    for mode in ['Baseline', 'GDCR']:
        avg_test_acc = np.mean([res['history']['test_acc'] for res in results[mode]], axis=0)
        plt.plot(avg_test_acc, label=mode)
    plt.title('Average Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gradient_direction_consistency_experiment/comparison.png')

    # Save summary results
    with open('gradient_direction_consistency_experiment/results.txt', 'w') as f:
        for mode in ['Baseline', 'GDCR']:
            accs = [res['best_test_acc'] for res in results[mode]]
            f.write(f"Mode: {mode}\n")
            f.write(f"  Test Accuracies: {accs}\n")
            f.write(f"  Mean: {np.mean(accs):.4f}, Std: {np.std(accs):.4f}\n")
            f.write("-" * 20 + "\n")
