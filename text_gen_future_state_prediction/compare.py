import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
from data import get_dataloaders
from model import GRULM, get_fsp_loss

# Hyperparameters
BATCH_SIZE = 64
SEQ_LENGTH = 64
EMBED_SIZE = 64
HIDDEN_SIZE = 128
NUM_EPOCHS = 3
SUBSET_SIZE = 10000
K_FUTURE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(lr, fsp_lambda, train_loader, val_loader, vocab_size):
    model = GRULM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, k_future=K_FUTURE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            logits, h_states, _ = model(x)

            # Reshape for CrossEntropy
            # logits: (batch, seq, vocab) -> (batch * seq, vocab)
            # y: (batch, seq) -> (batch * seq)
            ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            if fsp_lambda > 0:
                fsp_loss = get_fsp_loss(h_states, model.fsp_head, K_FUTURE)
                loss = ce_loss + fsp_lambda * fsp_loss
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()
            total_train_loss += ce_loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    print(f"Trial finished. Mode: {fsp_lambda > 0}, LR: {lr:.4f}, Lambda: {fsp_lambda:.4f}, Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss

def objective(trial, mode, train_loader, val_loader, vocab_size):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if mode == "experiment":
        fsp_lambda = trial.suggest_float("fsp_lambda", 0.01, 1.0, log=True)
    else:
        fsp_lambda = 0.0

    return train_model(lr, fsp_lambda, train_loader, val_loader, vocab_size)

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    train_loader, val_loader, vocab_size, char_to_ix, ix_to_char = get_dataloaders(BATCH_SIZE, SEQ_LENGTH, SUBSET_SIZE)

    print("Running Baseline Study...")
    baseline_study = optuna.create_study(direction="minimize")
    baseline_study.optimize(lambda t: objective(t, "baseline", train_loader, val_loader, vocab_size), n_trials=4, timeout=120)

    print("Running Experiment Study...")
    experiment_study = optuna.create_study(direction="minimize")
    experiment_study.optimize(lambda t: objective(t, "experiment", train_loader, val_loader, vocab_size), n_trials=4, timeout=120)

    print("\nResults:")
    print(f"Baseline Best Val Loss: {baseline_study.best_value:.4f}")
    print(f"Baseline Best Params: {baseline_study.best_params}")

    print(f"Experiment Best Val Loss: {experiment_study.best_value:.4f}")
    print(f"Experiment Best Params: {experiment_study.best_params}")

    # Final training with best parameters to get curves
    train_loader, val_loader, vocab_size, char_to_ix, ix_to_char = get_dataloaders(BATCH_SIZE, SEQ_LENGTH, SUBSET_SIZE)

    def get_curves(lr, fsp_lambda):
        model = GRULM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, k_future=K_FUTURE).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_losses = []

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_train_loss = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits, h_states, _ = model(x)
                ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                if fsp_lambda > 0:
                    fsp_loss = get_fsp_loss(h_states, model.fsp_head, K_FUTURE)
                    loss = ce_loss + fsp_lambda * fsp_loss
                else:
                    loss = ce_loss
                loss.backward()
                optimizer.step()
                total_train_loss += ce_loss.item()

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits, _, _ = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    total_val_loss += loss.item()

            train_losses.append(total_train_loss / len(train_loader))
            val_losses.append(total_val_loss / len(val_loader))
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        return train_losses, val_losses, model

    print("\nTraining Final Baseline Model...")
    NUM_EPOCHS = 5 # More epochs for final comparison
    b_train, b_val, b_model = get_curves(baseline_study.best_params['lr'], 0.0)

    print("\nTraining Final Experiment Model...")
    e_train, e_val, e_model = get_curves(experiment_study.best_params['lr'], experiment_study.best_params['fsp_lambda'])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(b_val, label="Baseline Val Loss")
    plt.plot(e_val, label="Experiment Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Baseline vs Future State Prediction Regularization")
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), "loss_comparison.png"))

    # Qualitative test
    def generate(model, start_str=None, length=100):
        model.eval()
        if start_str is None:
            # Pick first characters from val_loader
            x, _ = next(iter(val_loader))
            chars = x[0, :10].tolist()
            start_str = "".join([ix_to_char[c] for c in chars])
        else:
            chars = [char_to_ix[c] for c in start_str]
        input = torch.tensor(chars).unsqueeze(0).to(DEVICE)
        h = None

        generated_str = start_str

        with torch.no_grad():
            # Initial pass to get hidden state
            logits, _, h = model(input)
            char_idx = torch.argmax(logits[0, -1, :]).item()
            generated_str += ix_to_char[char_idx]

            for _ in range(length):
                input = torch.tensor([[char_idx]]).to(DEVICE)
                logits, _, h = model(input, h)
                char_idx = torch.argmax(logits[0, -1, :]).item()
                generated_str += ix_to_char[char_idx]

        return generated_str

    print("\nBaseline Generation:")
    print(generate(b_model))
    print("\nExperiment Generation:")
    print(generate(e_model))

    with open(os.path.join(os.path.dirname(__file__), "results.txt"), "w") as f:
        f.write(f"Baseline Best Val Loss: {baseline_study.best_value:.4f}\n")
        f.write(f"Experiment Best Val Loss: {experiment_study.best_value:.4f}\n")
        f.write("\nBaseline Generation:\n")
        f.write(generate(b_model) + "\n")
        f.write("\nExperiment Generation:\n")
        f.write(generate(e_model) + "\n")
