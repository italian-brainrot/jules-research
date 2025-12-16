import torch
import torch.nn as nn
import torch.optim as optim
from mnist1d.data import get_dataset, get_dataset_args
from .model import MLP
from light_dataloader import TensorDataLoader as DataLoader
import numpy as np
import os
import copy

BATCH_SIZE = 128

def get_model_and_data():
    """Initializes the model and data loaders."""
    args = get_dataset_args()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, 'mnist1d_data.pkl')
    data = get_dataset(args, path=data_path)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    train_loader = DataLoader((X_train, y_train), batch_size=BATCH_SIZE)
    val_loader = DataLoader((X_test, y_test), batch_size=BATCH_SIZE)

    input_size = X_train.shape[-1]
    output_size = len(torch.unique(y_train))
    model = MLP(input_size, output_size)
    return model, train_loader, val_loader

def train_and_evaluate(model_template, train_loader, val_loader, schedule_func, n_epochs):
    """Trains and evaluates a model with a given LR schedule."""
    model = copy.deepcopy(model_template)
    optimizer = optim.Adam(model.parameters(), lr=1.0)
    criterion = nn.CrossEntropyLoss()
    val_accuracies = []

    for epoch in range(n_epochs):
        epoch_normalized = epoch / (n_epochs - 1) if n_epochs > 1 else 0
        lr = schedule_func(epoch_normalized)
        lr = np.clip(lr, 1e-7, 1e-1)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        val_accuracies.append(accuracy)

    return val_accuracies
