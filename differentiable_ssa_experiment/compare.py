import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from ssa import SSANet
import matplotlib.pyplot as plt
import sys

class BaselineMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_noisy_mnist1d(noise_std=0.2, seed=42):
    torch.manual_seed(seed)
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y'])
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test'])

    # Add noise to training and test sets
    X_train += noise_std * torch.randn_like(X_train)
    X_test += noise_std * torch.randn_like(X_test)

    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, dl_test, lr, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in dl_train:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_test:
                out = model(x)
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        acc = 100 * correct / total
        history.append(acc)
    return acc, history

def run_experiment(lr_baseline, lr_ssa, window_size, seeds=[42, 43, 44, 45, 46]):
    baseline_accs = []
    ssa_accs = []

    for seed in seeds:
        print(f"Running seed {seed}...")
        X_train, y_train, X_test, y_test = get_noisy_mnist1d(seed=seed)
        dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
        dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

        # Baseline
        torch.manual_seed(seed)
        model_b = BaselineMLP(40, 128, 10)
        acc_b, _ = train_model(model_b, dl_train, dl_test, lr_baseline)
        baseline_accs.append(acc_b)

        # SSA
        torch.manual_seed(seed)
        model_s = SSANet(40, window_size, 128, 10)
        acc_s, _ = train_model(model_s, dl_train, dl_test, lr_ssa)
        ssa_accs.append(acc_s)

        print(f"Seed {seed}: Baseline {acc_b:.2f}%, SSA {acc_s:.2f}%")

    return baseline_accs, ssa_accs

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 compare.py <lr_baseline> <lr_ssa> <window_size>")
        sys.exit(1)

    lr_baseline = float(sys.argv[1])
    lr_ssa = float(sys.argv[2])
    window_size = int(sys.argv[3])

    baseline_accs, ssa_accs = run_experiment(lr_baseline, lr_ssa, window_size)

    print("\nFinal Results:")
    print(f"Baseline: {np.mean(baseline_accs):.2f}% ± {np.std(baseline_accs):.2f}%")
    print(f"SSA:      {np.mean(ssa_accs):.2f}% ± {np.std(ssa_accs):.2f}%")

    with open("differentiable_ssa_experiment/results.txt", "w") as f:
        f.write(f"Baseline: {np.mean(baseline_accs):.2f}% ± {np.std(baseline_accs):.2f}%\n")
        f.write(f"SSA:      {np.mean(ssa_accs):.2f}% ± {np.std(ssa_accs):.2f}%\n")
        f.write(f"Baseline List: {baseline_accs}\n")
        f.write(f"SSA List: {ssa_accs}\n")
