import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import math
import os
import json
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from layers import HarmonicParameterizedEmbedding, FactorizedEmbedding, StandardEmbedding

# Hyperparameters for the task
Q = 256 # Quantization levels
D = 64
N_CLASSES = 10
SEQ_LEN = 40

# Cache data
DATA_CACHE = None

def get_data():
    global DATA_CACHE
    if DATA_CACHE is not None:
        return DATA_CACHE

    defaults = get_dataset_args()
    defaults.num_samples = 4000 # Back to default or slightly more
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x']), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]), torch.tensor(data['y_test'])

    # Quantize
    v_min, v_max = -5.0, 5.0
    X_train_q = torch.clamp((X_train - v_min) / (v_max - v_min), 0, 1)
    X_train_q = (X_train_q * (Q - 1)).long()

    X_test_q = torch.clamp((X_test - v_min) / (v_max - v_min), 0, 1)
    X_test_q = (X_test_q * (Q - 1)).long()

    DATA_CACHE = (X_train_q, y_train, X_test_q, y_test)
    return DATA_CACHE

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_layer, d_model, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        self.embedding = embedding_layer
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

def train_model(model, dl_train, dl_test, lr, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in dl_train:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_test:
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
    return best_acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def objective_method(trial, method):
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)

    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

    if method == "standard":
        emb = StandardEmbedding(Q, D)
    elif method == "factorized":
        K = trial.suggest_int("factor_K", 8, 48)
        emb = FactorizedEmbedding(Q, D, K)
    elif method == "hpe":
        K = trial.suggest_int("hpe_K", 4, 16)
        emb = HarmonicParameterizedEmbedding(Q, D, K)

    model = TransformerClassifier(emb, D)
    params = count_parameters(model)
    trial.set_user_attr("params", params)

    acc = train_model(model, dl_train, dl_test, lr, epochs=10)
    return acc

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    methods = ["standard", "factorized", "hpe"]
    results = {}

    for method in methods:
        print(f"Optimizing {method}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective_method(t, method), n_trials=8)

        results[method] = {
            "best_acc": study.best_value,
            "best_params": study.best_params,
            "params": study.best_trial.user_attrs["params"]
        }
        print(f"Best {method} acc: {study.best_value} (params: {results[method]['params']})")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
