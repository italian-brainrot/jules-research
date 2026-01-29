import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
from neural_count_vectorizer_experiment.data_utils import get_data, collate_fn
from neural_count_vectorizer_experiment.models import *
from torch.utils.data import DataLoader

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            y = torch.stack([item[3] for item in batch]).to(device)
            logits = model(batch)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total, total_loss / total

def train_full(model, train_loader, val_loader, epochs=30, lr=1e-3, weight_decay=1e-5, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            y = torch.stack([item[3] for item in batch]).to(device)
            logits = model(batch)
            if logits is None: continue
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_acc, _ = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Early stopping if perfect accuracy reached
        if val_acc == 1.0:
            break

    model.load_state_dict(best_state)
    return model, best_val_acc

def objective(trial, weighting_name, datasets, stats, device):
    embedding_dim = 128
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    if weighting_name == "Uniform":
        scheme = UniformWeighting()
    elif weighting_name == "TF":
        scheme = TFWeighting()
    elif weighting_name == "LogTFIDF":
        scheme = LogTFIDFWeighting()
    elif weighting_name == "BM25":
        scheme = BM25Weighting()
    elif weighting_name == "NCV":
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
        scheme = NCVWeighting(hidden_dim=hidden_dim)

    model = BOWClassifier(
        vocab_size=stats['vocab_size'],
        num_classes=4,
        embedding_dim=embedding_dim,
        weighting_scheme=scheme,
        df=stats['df'],
        avg_doc_len=stats['avg_doc_length'],
        num_docs=stats['num_docs']
    )

    train_loader = DataLoader(datasets['train'], batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(datasets['val'], batch_size=64, shuffle=False, collate_fn=collate_fn)

    _, best_val_acc = train_full(model, train_loader, val_loader, epochs=10, lr=lr, weight_decay=weight_decay, device=device)
    return best_val_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets, stats = get_data()

    results = {}
    weighting_schemes = ["Uniform", "TF", "LogTFIDF", "BM25", "NCV"]

    for name in weighting_schemes:
        print(f"\n--- Tuning {name} ---")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, name, datasets, stats, device), n_trials=5)

        print(f"Best params for {name}: {study.best_params}")

        # Final training with best params
        print(f"Final training for {name}...")
        if name == "Uniform":
            scheme = UniformWeighting()
        elif name == "TF":
            scheme = TFWeighting()
        elif name == "LogTFIDF":
            scheme = LogTFIDFWeighting()
        elif name == "BM25":
            scheme = BM25Weighting()
        elif name == "NCV":
            scheme = NCVWeighting(hidden_dim=study.best_params['hidden_dim'])

        model = BOWClassifier(
            vocab_size=stats['vocab_size'],
            num_classes=4,
            embedding_dim=128,
            weighting_scheme=scheme,
            df=stats['df'],
            avg_doc_len=stats['avg_doc_length'],
            num_docs=stats['num_docs']
        )

        train_loader = DataLoader(datasets['train'], batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(datasets['val'], batch_size=64, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(datasets['test'], batch_size=64, shuffle=False, collate_fn=collate_fn)

        model, _ = train_full(model, train_loader, val_loader, epochs=20,
                             lr=study.best_params['lr'],
                             weight_decay=study.best_params['weight_decay'],
                             device=device)

        test_acc, test_loss = evaluate(model, test_loader, device)
        print(f"Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        results[name] = {"acc": test_acc, "loss": test_loss, "params": study.best_params}

        if name == "NCV":
            # Save NCV analysis data
            analyze_ncv(model, stats, device)

    # Print summary
    print("\n--- Summary Results ---")
    for name, res in results.items():
        print(f"{name:10}: Acc={res['acc']:.4f}, Loss={res['loss']:.4f}")

def analyze_ncv(model, stats, device):
    model.eval()
    with torch.no_grad():
        # Test range of tf and df
        tf_vals = torch.linspace(0, 50, 100, dtype=torch.float32).to(device)
        df_vals = torch.linspace(0, float(stats['num_docs']), 100, dtype=torch.float32).to(device)
        avg_doc_len = float(stats['avg_doc_length'])
        num_docs = stats['num_docs']

        # Fixed df, varying tf
        fixed_df = torch.tensor([stats['num_docs'] // 10], dtype=torch.float32).to(device).expand(100)
        fixed_doc_len = torch.tensor([avg_doc_len], dtype=torch.float32).to(device).expand(100)
        weights_tf = model.weighting_scheme(tf_vals, fixed_df, fixed_doc_len, avg_doc_len, num_docs)

        # Fixed tf, varying df
        fixed_tf = torch.tensor([1.0], dtype=torch.float32).to(device).expand(100)
        weights_df = model.weighting_scheme(fixed_tf, df_vals, fixed_doc_len, avg_doc_len, num_docs)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(tf_vals.cpu().numpy(), weights_tf.cpu().numpy())
        plt.xlabel("Term Frequency (tf)")
        plt.ylabel("Learned Weight")
        plt.title("Weight vs TF (fixed df)")

        plt.subplot(1, 2, 2)
        plt.plot(df_vals.cpu().numpy(), weights_df.cpu().numpy())
        plt.xlabel("Document Frequency (df)")
        plt.ylabel("Learned Weight")
        plt.title("Weight vs DF (fixed tf)")

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "ncv_analysis.png"))
        print("Saved NCV analysis plot.")

if __name__ == "__main__":
    main()
