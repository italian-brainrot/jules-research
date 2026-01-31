import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from light_dataloader import TensorDataLoader

def compute_cov_loss(h1, h2):
    # h1: (B, D), h2: (B, D)
    b = h1.shape[0]
    h1_centered = h1 - h1.mean(dim=0, keepdim=True)
    h2_centered = h2 - h2.mean(dim=0, keepdim=True)
    cov = (h1_centered.t() @ h2_centered) / (b - 1)
    return torch.norm(cov, p='fro')**2

def train_backprop(model, dl_train, lr, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for bx, by in dl_train:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
    return model

def train_greedy_stacking(model, dl_train, lr, epochs_per_layer, device):
    for i in range(model.num_layers):
        # Parameters to optimize: only the i-th layer and its head
        params = list(model.layers[i].parameters()) + list(model.heads[i].parameters())
        optimizer = optim.Adam(params, lr=lr)

        # Freeze all layers
        for p in model.parameters():
            p.requires_grad = False
        for p in params:
            p.requires_grad = True

        model.train()
        for epoch in range(epochs_per_layer):
            for bx, by in dl_train:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()

                # Manual forward to avoid recomputing everything if we wanted to be efficient,
                # but for simplicity, use the model's methods.
                logits = model.forward_head(bx, i)
                loss = F.cross_entropy(logits, by)
                loss.backward()
                optimizer.step()
    return model

def train_greedy_boosting(model, dl_train, lr, epochs_per_layer, device):
    for i in range(model.num_layers):
        params = list(model.layers[i].parameters()) + list(model.heads[i].parameters())
        optimizer = optim.Adam(params, lr=lr)

        for p in model.parameters():
            p.requires_grad = False
        for p in params:
            p.requires_grad = True

        model.train()
        for epoch in range(epochs_per_layer):
            for bx, by in dl_train:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()

                logits = model.forward_boost(bx, i)
                loss = F.cross_entropy(logits, by)
                loss.backward()
                optimizer.step()
    return model

def train_greedy_boosting_fd(model, dl_train, lr, lambd, epochs_per_layer, device):
    for i in range(model.num_layers):
        params = list(model.layers[i].parameters()) + list(model.heads[i].parameters())
        optimizer = optim.Adam(params, lr=lr)

        for p in model.parameters():
            p.requires_grad = False
        for p in params:
            p.requires_grad = True

        model.train()
        for epoch in range(epochs_per_layer):
            for bx, by in dl_train:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()

                # Get current and previous hidden states
                h_prev = bx
                if i > 0:
                    with torch.no_grad():
                        h_prev = model.forward_layer(bx, i-1)

                h_curr = model.layers[i](h_prev)

                # Get logits
                with torch.no_grad():
                    logits_prev = 0
                    if i > 0:
                        logits_prev = model.forward_boost(bx, i-1)

                logits_curr = logits_prev + model.heads[i](h_curr)

                loss_ce = F.cross_entropy(logits_curr, by)

                loss_fd = 0
                if i > 0 and lambd > 0:
                    loss_fd = compute_cov_loss(h_curr, h_prev)

                loss = loss_ce + lambd * loss_fd
                loss.backward()
                optimizer.step()
    return model

def evaluate(model, X_test, y_test, device, method='backprop'):
    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)
    with torch.no_grad():
        if method == 'backprop' or method == 'stacking':
            logits = model(X_test)
        else: # boosting
            logits = model.forward_all_heads_sum(X_test)

        preds = logits.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc
