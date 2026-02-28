import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import vmap, grad, functional_call
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import MLP
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)
    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Split train into train and val
    n_train = int(0.8 * len(x_train))
    x_val = x_train[n_train:]
    y_val = y_train[n_train:]
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    return x_train, y_train, x_val, y_val, x_test, y_test

def compute_loss_single(model, params, buffers, x_single, y_single):
    logits = functional_call(model, (params, buffers), (x_single.unsqueeze(0),))
    loss = F.cross_entropy(logits, y_single.unsqueeze(0))
    return loss

def train_epoch(model, loader, optimizer, criterion, mode, lambda_igsr=0.0):
    model.train()
    total_loss = 0

    # Pre-bind grad function for IGSR
    if mode == 'IGSR':
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        grad_fn = vmap(grad(compute_loss_single, argnums=3), in_dims=(None, None, None, 0, 0))

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        output = model(x)
        ce_loss = criterion(output, y)

        if mode == 'IGSR':
            # Compute per-sample input gradients
            input_grads = grad_fn(model, params, buffers, x, y)
            # IGSR penalty: mean over batch and spatial diff squared
            diffs = input_grads[:, 1:] - input_grads[:, :-1]
            igsr_loss = lambda_igsr * torch.mean(diffs**2)
            loss = ce_loss + igsr_loss
        else:
            loss = ce_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += criterion(output, y).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return total_loss / len(loader), correct / loader.data_length()

def objective(trial, mode, data):
    set_seed(42)
    x_train, y_train, x_val, y_val = data
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    model = MLP().to(device)

    if mode == 'IGSR':
        lambda_igsr = trial.suggest_float('lambda_igsr', 1e-5, 10.0, log=True)
    else:
        lambda_igsr = 0.0

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(20):
        train_epoch(model, train_loader, optimizer, criterion, mode, lambda_igsr)
        _, val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def run_experiment():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data_for_obj = (x_train, y_train, x_val, y_val)

    modes = ['Baseline', 'IGSR']
    best_params = {}

    for mode in modes:
        print(f"Optimizing {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data_for_obj), n_trials=30)
        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {study.best_params}")

    # Final training and evaluation
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    results = {}
    seeds = [42, 43, 44]

    for mode in modes:
        mode_results = []
        print(f"Final training for {mode}...")
        params = best_params[mode]
        lambda_igsr = params.get('lambda_igsr', 0.0)

        for seed in seeds:
            set_seed(seed)
            model = MLP().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            criterion = nn.CrossEntropyLoss()

            history = {'train_loss': [], 'val_acc': [], 'test_acc': []}
            best_val_acc = 0
            test_acc_at_best_val = 0

            for epoch in range(50):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, mode, lambda_igsr)
                val_loss, val_acc = evaluate(model, val_loader)
                test_loss, test_acc = evaluate(model, test_loader)

                history['train_loss'].append(train_loss)
                history['val_acc'].append(val_acc)
                history['test_acc'].append(test_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc_at_best_val = test_acc

            mode_results.append({
                'test_acc': test_acc_at_best_val,
                'history': history,
                'model_state': model.state_dict() # For saliency visualization later
            })
            print(f"Seed {seed}, Test Acc: {test_acc_at_best_val:.4f}")

        results[mode] = {
            'test_accs': [r['test_acc'] for r in mode_results],
            'mean': np.mean([r['test_acc'] for r in mode_results]),
            'std': np.std([r['test_acc'] for r in mode_results]),
            'best_params': params,
            'histories': [r['history'] for r in mode_results],
            'last_model_state': mode_results[0]['model_state']
        }

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for mode in modes:
        avg_train_loss = np.mean([h['train_loss'] for h in results[mode]['histories']], axis=0)
        plt.plot(avg_train_loss, label=mode)
    plt.title('Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in modes:
        avg_val_acc = np.mean([h['val_acc'] for h in results[mode]['histories']], axis=0)
        plt.plot(avg_val_acc, label=mode)
    plt.title('Average Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('input_gradient_smoothness_experiment/results.png')
    plt.close()

    # Visualization of saliency maps
    visualize_saliency(results, x_test[:10], y_test[:10])

    # Save README
    save_readme(results, modes)

def visualize_saliency(results, x_samples, y_samples):
    x_samples = x_samples.to(device)
    y_samples = y_samples.to(device)

    plt.figure(figsize=(15, 6))
    for i, mode in enumerate(['Baseline', 'IGSR']):
        model = MLP().to(device)
        model.load_state_dict(results[mode]['last_model_state'])
        model.eval()

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        grad_fn = vmap(grad(compute_loss_single, argnums=3), in_dims=(None, None, None, 0, 0))

        input_grads = grad_fn(model, params, buffers, x_samples, y_samples)
        input_grads = input_grads.cpu().detach().numpy()

        for j in range(5):
            plt.subplot(2, 5, i*5 + j + 1)
            plt.plot(input_grads[j])
            if j == 0:
                plt.ylabel(f'{mode} Saliency')
            if i == 0:
                plt.title(f'Sample {j}')

    plt.tight_layout()
    plt.savefig('input_gradient_smoothness_experiment/saliency_maps.png')
    plt.close()

def save_readme(results, modes):
    with open('input_gradient_smoothness_experiment/README.md', 'w') as f:
        f.write("# Input Gradient Smoothness Regularization (IGSR) Experiment\n\n")
        f.write("## Hypothesis\n")
        f.write("For 1D sequential or spatial data, the importance of input features should be spatially coherent. ")
        f.write("By penalizing the difference between adjacent elements in the input gradient (saliency map), ")
        f.write("we encourage the model to focus on spatially continuous structures and ignore high-frequency noise.\n\n")
        f.write("Penalty: $L_{IGSR} = \\lambda \\cdot \\text{mean}((\\nabla_{x_{i+1}} L - \\nabla_{x_i} L)^2)$\n\n")

        f.write("## Results\n")
        f.write("| Mode | Test Accuracy | Best Hyperparameters |\n")
        f.write("| --- | --- | --- |\n")
        for mode in modes:
            res = results[mode]
            f.write(f"| {mode} | {res['mean']:.4f} ± {res['std']:.4f} | {res['best_params']} |\n")

        f.write("\n## Visualizations\n")
        f.write("### Training and Validation Curves\n")
        f.write("![Results](results.png)\n\n")
        f.write("### Saliency Maps\n")
        f.write("The following plot shows the input gradients for 5 samples from the test set for both models. ")
        f.write("IGSR is expected to produce smoother saliency maps.\n\n")
        f.write("![Saliency Maps](saliency_maps.png)\n\n")

        f.write("## Discussion\n")
        if results['IGSR']['mean'] > results['Baseline']['mean']:
            f.write("IGSR outperformed the baseline, suggesting that encouraging spatial coherence in input sensitivity improves generalization.\n")
        else:
            f.write("IGSR did not outperform the baseline in this setup. This could be due to the nature of the mnist1d dataset or the choice of the penalty weight.\n")

if __name__ == '__main__':
    run_experiment()
