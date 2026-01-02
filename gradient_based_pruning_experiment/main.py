import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import torch.nn.utils.prune as prune

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Load the mnist1d dataset
defaults = get_dataset_args()
defaults.num_samples = 10000
data = make_dataset(defaults)
X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])
dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

def train(model, dataloader, criterion, optimizer, grad_accumulators=None, beta=0.99):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if grad_accumulators is not None:
            for name, param in model.named_parameters():
                if 'weight' in name:
                    grad_accumulators[name] = beta * grad_accumulators[name] + (1 - beta) * torch.abs(param.grad)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

def prune_by_gradient_magnitude(model, grad_accumulators, amount):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear)):
            if f"{name}.weight" in grad_accumulators:
                tensor = grad_accumulators[f"{name}.weight"]
                prune.l1_unstructured(module, name="weight", amount=amount, importance_scores=tensor)
                prune.remove(module, 'weight')

def prune_by_magnitude(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, 'weight')

def run_experiment(pruning_method, pruning_amount, lr, epochs, pruning_epoch, input_size, hidden_size, output_size):
    torch.manual_seed(0)
    np.random.seed(0)

    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    grad_accumulators = None
    if pruning_method == 'gradient':
        grad_accumulators = {name: torch.zeros_like(param) for name, param in model.named_parameters() if 'weight' in name}

    for epoch in range(epochs):
        train(model, dl_train, criterion, optimizer, grad_accumulators=grad_accumulators)
        if epoch == pruning_epoch:
            if pruning_method == 'gradient':
                prune_by_gradient_magnitude(model, grad_accumulators, pruning_amount)
            elif pruning_method == 'magnitude':
                prune_by_magnitude(model, pruning_amount)

    return evaluate(model, dl_test)

def main():
    input_size = 40
    hidden_size = 64
    output_size = 10
    epochs = 20
    pruning_epoch = 10
    lr = 0.01

    pruning_amounts = np.arange(0.1, 1.0, 0.1)
    grad_accuracies = []
    mag_accuracies = []

    for amount in pruning_amounts:
        print(f"Running experiments for pruning amount: {amount:.1f}")

        grad_acc = run_experiment('gradient', amount, lr, epochs, pruning_epoch, input_size, hidden_size, output_size)
        grad_accuracies.append(grad_acc)
        print(f"  Gradient-based pruning accuracy: {grad_acc:.4f}")

        mag_acc = run_experiment('magnitude', amount, lr, epochs, pruning_epoch, input_size, hidden_size, output_size)
        mag_accuracies.append(mag_acc)
        print(f"  Magnitude-based pruning accuracy: {mag_acc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(pruning_amounts, grad_accuracies, marker='o', label='Gradient-based Pruning')
    plt.plot(pruning_amounts, mag_accuracies, marker='x', label='Magnitude-based Pruning')
    plt.title('Pruning Method Comparison')
    plt.xlabel('Sparsity (Pruning Amount)')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('gradient_based_pruning_experiment/pruning_comparison.png')
    print("Plot saved to gradient_based_pruning_experiment/pruning_comparison.png")

if __name__ == '__main__':
    main()
