import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from supervised_gradient_alignment_experiment.train import MLP, train_epoch, evaluate, get_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_evaluation(n_seeds=3):
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    results_dir = 'supervised_gradient_alignment_experiment'

    with open(os.path.join(results_dir, 'best_params_Baseline.json'), 'r') as f:
        baseline_params = json.load(f)

    with open(os.path.join(results_dir, 'best_params_SGA.json'), 'r') as f:
        sga_params = json.load(f)

    all_results = {'Baseline': [], 'SGA': []}

    for mode, params in [('Baseline', baseline_params), ('SGA', sga_params)]:
        print(f"Evaluating {mode}...")
        for seed in range(n_seeds):
            set_seed(seed)
            model = MLP().to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )

            lambda_intra = params.get('lambda_intra', 0.0)
            lambda_inter = params.get('lambda_inter', 0.0)

            for epoch in range(25):
                train_epoch(model, train_loader, optimizer, mode, lambda_intra, lambda_inter, device)

            acc = evaluate(model, test_loader, device)
            all_results[mode].append(acc)
            print(f"Seed {seed}, Accuracy: {acc:.4f}")

    # Save results
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f)

    # Calculate statistics
    for mode in all_results:
        accs = np.array(all_results[mode])
        print(f"{mode}: {accs.mean():.4f} +/- {accs.std():.4f}")

    # Plot results
    plt.figure(figsize=(8, 6))
    modes = list(all_results.keys())
    data = [all_results[m] for m in modes]
    plt.boxplot(data, labels=modes)
    plt.ylabel('Test Accuracy')
    plt.title('Baseline vs Supervised Gradient Alignment (SGA)')
    plt.savefig(os.path.join(results_dir, 'comparison_plot.png'))
    plt.close()

if __name__ == "__main__":
    run_evaluation()
