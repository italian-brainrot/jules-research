import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import optuna
import os
import matplotlib.pyplot as plt
import copy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    # The dataset returns float64 numpy arrays. PyTorch defaults to float32.
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=100, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = F.cross_entropy(outputs, y).item()
        preds = outputs.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return loss, acc

def train_model(method, batch_size, epochs=30, alpha_0=0.1):
    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    model = MLP()
    criterion_mean = nn.CrossEntropyLoss(reduction='mean')
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    beta = 0.5
    max_backtrack_steps = 10

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'alphas': []}

    g_prev = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        epoch_alphas = []

        for inputs, targets in dl_train:
            if method == 'plain':
                # 1. Compute current gradient
                model.zero_grad()
                outputs = model(inputs)
                loss = criterion_mean(outputs, targets)
                loss.backward()
                grads = [p.grad.clone() for p in model.parameters()]

                # 2. Backtracking on mean loss
                orig_params = [copy.deepcopy(p.data) for p in model.parameters()]
                f_0 = loss.item()

                best_alpha = 0
                alpha = alpha_0
                for _ in range(max_backtrack_steps):
                    with torch.no_grad():
                        for p, g in zip(model.parameters(), grads):
                            p.data.sub_(alpha * g)

                        f_alpha = criterion_mean(model(inputs), targets).item()

                    if f_alpha < f_0:
                        best_alpha = alpha
                        break

                    # Restore
                    with torch.no_grad():
                        for p, orig in zip(model.parameters(), orig_params):
                            p.data.copy_(orig)
                    alpha *= beta

                epoch_loss += f_0
                epoch_alphas.append(best_alpha)

            elif method == 'variant':
                if g_prev is None:
                    # Initial gradient
                    model.zero_grad()
                    outputs = model(inputs)
                    loss = criterion_mean(outputs, targets)
                    loss.backward()
                    g_prev = [p.grad.clone() for p in model.parameters()]
                    epoch_loss += loss.item()
                    num_batches += 1
                    continue

                # 1. Backtracking using g_prev on current batch to minimize max loss
                orig_params = [copy.deepcopy(p.data) for p in model.parameters()]
                with torch.no_grad():
                    f_0 = criterion_none(model(inputs), targets).max().item()

                best_alpha = 0
                alpha = alpha_0
                for _ in range(max_backtrack_steps):
                    with torch.no_grad():
                        for p, g in zip(model.parameters(), g_prev):
                            p.data.sub_(alpha * g)

                        f_alpha = criterion_none(model(inputs), targets).max().item()

                    if f_alpha < f_0:
                        best_alpha = alpha
                        break

                    # Restore
                    with torch.no_grad():
                        for p, orig in zip(model.parameters(), orig_params):
                            p.data.copy_(orig)
                    alpha *= beta

                # 2. Compute g_prev for NEXT step at NEW point (or same point if best_alpha=0)
                model.zero_grad()
                outputs = model(inputs)
                loss = criterion_mean(outputs, targets)
                loss.backward()
                g_prev = [p.grad.clone() for p in model.parameters()]

                epoch_loss += loss.item()
                epoch_alphas.append(best_alpha)

            num_batches += 1

        val_loss, val_acc = evaluate(model, X_test, y_test)
        history['train_loss'].append(epoch_loss / num_batches)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['alphas'].append(np.mean(epoch_alphas))
        logger.info(f"Epoch {epoch}: {method} - Val Acc: {val_acc:.4f}, Avg Alpha: {np.mean(epoch_alphas):.4f}")

    return history, model

def run_experiment():
    results = {}
    methods = ['plain', 'variant']
    alpha_0 = 0.5 # Fixed initial step size

    for method in methods:
        logger.info(f"Tuning batch size for {method}...")

        def objective(trial):
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
            history, _ = train_model(method, batch_size, epochs=15, alpha_0=alpha_0)
            return max(history['val_acc'])

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10) # 10 trials as suggested by memory

        best_batch_size = study.best_params['batch_size']
        logger.info(f"Best batch size for {method}: {best_batch_size}")

        logger.info(f"Running final training for {method}...")
        history, model = train_model(method, best_batch_size, epochs=50, alpha_0=alpha_0)
        results[method] = {
            'history': history,
            'best_batch_size': best_batch_size
        }

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for method in methods:
        plt.plot(results[method]['history']['val_acc'], label=f"{method} (bs={results[method]['best_batch_size']})")
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    for method in methods:
        plt.plot(results[method]['history']['val_loss'], label=f"{method} (bs={results[method]['best_batch_size']})")
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plot_path = os.path.join(os.path.dirname(__file__), 'results.png')
    plt.savefig(plot_path)
    logger.info(f"Saved plot to {plot_path}")

    # Save Alpha history
    plt.figure()
    for method in methods:
        plt.plot(results[method]['history']['alphas'], label=method)
    plt.title('Average Alpha (Step Size) per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Alpha')
    plt.legend()
    alpha_plot_path = os.path.join(os.path.dirname(__file__), 'alphas.png')
    plt.savefig(alpha_plot_path)

    return results

if __name__ == "__main__":
    results = run_experiment()

    # Write README.md
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Stochastic Backtracking with Max Loss Experiment\n\n")
        f.write("## Hypothesis\n")
        f.write("A variant of backtracking line search for SGD that uses the gradient from the previous mini-batch and minimizes the maximum per-sample loss on the current mini-batch will provide better robustness and potentially better generalization than standard backtracking SGD on the mean loss.\n\n")

        f.write("## Methodology\n")
        f.write("- **Dataset**: MNIST-1D (10,000 samples).\n")
        f.write("- **Model**: Simple MLP (40 -> 100 -> 100 -> 10).\n")
        f.write("- **Plain Method**: Backtracking line search on mean loss using current batch for direction and step size.\n")
        f.write("- **Variant Method**: Backtracking line search on max per-sample loss using previous batch gradient for direction.\n")
        f.write("- **Tuning**: Batch size tuned for both methods using Optuna (10 trials each, range [32, 64, 128, 256]).\n")
        f.write("- **Fixed LR**: Initial step size `alpha_0 = 0.5` for both.\n\n")

        f.write("## Results\n")
        f.write(f"- **Plain SGD Best Batch Size**: {results['plain']['best_batch_size']}\n")
        f.write(f"- **Variant SGD Best Batch Size**: {results['variant']['best_batch_size']}\n")
        f.write(f"- **Plain SGD Best Val Acc**: {max(results['plain']['history']['val_acc']):.4f}\n")
        f.write(f"- **Variant SGD Best Val Acc**: {max(results['variant']['history']['val_acc']):.4f}\n\n")

        f.write("![Validation Results](results.png)\n")
        f.write("![Alpha History](alphas.png)\n\n")

        f.write("## Conclusion\n")
        plain_acc = max(results['plain']['history']['val_acc'])
        variant_acc = max(results['variant']['history']['val_acc'])
        if variant_acc > plain_acc:
            f.write("The variant method outperformed the plain method in terms of validation accuracy.\n")
        else:
            f.write("The plain method outperformed the variant method in terms of validation accuracy.\n")
