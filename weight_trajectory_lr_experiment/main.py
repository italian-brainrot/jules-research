import torch
import torch.nn as nn
import copy
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args

# 1. Model Definition
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. Optimizer Wrapper
class WeightTrajectoryLR:
    def __init__(self, optimizer, beta=1.0):
        self.optimizer = optimizer
        self.beta = beta
        # Store initial parameters for each group
        self.initial_params = [
            [p.clone().detach() for p in group['params']]
            for group in self.optimizer.param_groups
        ]
        self.original_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, closure=None):
        # Apply scaling for each parameter group
        for i, (group, initial_group_params) in enumerate(zip(self.optimizer.param_groups, self.initial_params)):
            # Iterate over the parameters within the group
            for j, p in enumerate(group['params']):
                # We only scale layers with weights (e.g., Linear, Conv), not biases
                if p.dim() > 1:
                    initial_p = initial_group_params[j]
                    dist = torch.norm(p.data - initial_p)
                    scaling_factor = 1.0 / (1.0 + self.beta * dist)
                    # Apply scaling to the group's LR. Note this affects all params in the group (weights and biases).
                    group['lr'] = self.original_lrs[i] * scaling_factor

        # Execute the optimizer step
        self.optimizer.step(closure=closure)

        # Restore original learning rates for the next iteration
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.original_lrs[i]


    def zero_grad(self):
        self.optimizer.zero_grad()

# 3. Training and Evaluation Functions
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# Main block for debugging
if __name__ == '__main__':
    # Hyperparameters
    input_size = 40
    hidden_size = 256
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 5
    batch_size = 128

    # Data Loading
    args = get_dataset_args()
    data = get_dataset(args, path='./mnist1d_data.pkl')
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.int64)
    X_test, y_test = torch.tensor(data['x_test'], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.int64)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=batch_size, shuffle=False)


    # Model, Optimizer, and Loss
    model = MLP(input_size, hidden_size, num_classes)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wrapped_optimizer = WeightTrajectoryLR(base_optimizer, beta=1.0)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, wrapped_optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
