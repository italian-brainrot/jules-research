import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import optuna
import os
import matplotlib.pyplot as plt
from model_gen import Generator
from executor import ProxyExecutor

def evaluate_rpn(expr, inputs):
    stack = []
    for token in expr:
        if token in ['a', 'b', 'c']:
            stack.append(inputs[token])
        elif token.replace('.','',1).isdigit():
            try:
                stack.append(float(token))
            except: return None
        else:
            if len(stack) < 2: return None
            v2 = stack.pop()
            v1 = stack.pop()
            try:
                if token == '+': stack.append(v1 + v2)
                elif token == '-': stack.append(v1 - v2)
                elif token == '*': stack.append(v1 * v2)
            except: return None
    return stack[0] if len(stack) == 1 else None

def objective(trial, mode, train_data, val_data, vocab, vocab_size, proxy, idx_to_token):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    lambda_def = 0.0
    if mode == 'def':
        lambda_def = trial.suggest_float("lambda_def", 1e-4, 0.5, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(vocab_size, 64, 128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0)
    criterion_exec = nn.MSELoss()

    batch_size = 128
    epochs = 10

    for epoch in range(epochs):
        random.shuffle(train_data)
        model.train()
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            if not batch: continue

            max_desc_len = max(len(b['desc']) for b in batch)
            descs = torch.stack([torch.nn.functional.pad(b['desc'], (0, max_desc_len - len(b['desc']))) for b in batch]).to(device)
            exprs = torch.stack([torch.nn.functional.pad(b['expr'], (0, 5 - len(b['expr']))) for b in batch]).to(device)

            optimizer.zero_grad()
            logits = model(descs, target_len=5, teacher_forcing=exprs)
            loss_ce = criterion_ce(logits.view(-1, vocab_size), exprs.view(-1))

            if mode == 'def':
                soft_exprs = model.forward_soft(descs, target_len=5, tau=1.0)
                inputs_list = []
                targets_list = []
                for b in batch:
                    io = random.choice(b['io'])
                    inputs_list.append(torch.tensor(io[0], dtype=torch.float32))
                    targets_list.append(torch.tensor(io[1], dtype=torch.float32))
                inputs = torch.stack(inputs_list).to(device)
                targets = torch.stack(targets_list).to(device)
                pred_outputs = proxy(soft_exprs, inputs)
                loss_exec = criterion_exec(pred_outputs, targets)
                loss = loss_ce + lambda_def * loss_exec
            else:
                loss = loss_ce

            loss.backward()
            optimizer.step()

    # Validation accuracy (functional)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for d in val_data:
            desc = d['desc'].unsqueeze(0).to(device)
            pred_logits = model(desc, target_len=5)
            pred_ids = torch.argmax(pred_logits, dim=-1).squeeze(0).cpu()

            expr_tokens = [idx_to_token[str(idx.item())] for idx in pred_ids if idx.item() >= 4]
            io = d['io'][0]
            res = evaluate_rpn(expr_tokens, {'a': io[0][0], 'b': io[0][1], 'c': io[0][2]})
            if res is not None and abs(res - io[1]) < 1e-3:
                correct += 1
            total += 1

    return correct / total

def train_final(mode, config, train_data, test_data, vocab, vocab_size, proxy, idx_to_token):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(vocab_size, 64, 128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0)
    criterion_exec = nn.MSELoss()

    batch_size = 128
    epochs = 20

    history = []
    for epoch in range(epochs):
        random.shuffle(train_data)
        model.train()
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            if not batch: continue
            max_desc_len = max(len(b['desc']) for b in batch)
            descs = torch.stack([torch.nn.functional.pad(b['desc'], (0, max_desc_len - len(b['desc']))) for b in batch]).to(device)
            exprs = torch.stack([torch.nn.functional.pad(b['expr'], (0, 5 - len(b['expr']))) for b in batch]).to(device)

            optimizer.zero_grad()
            logits = model(descs, target_len=5, teacher_forcing=exprs)
            loss_ce = criterion_ce(logits.view(-1, vocab_size), exprs.view(-1))

            if mode == 'def':
                soft_exprs = model.forward_soft(descs, target_len=5, tau=1.0)
                inputs_list = []
                targets_list = []
                for b in batch:
                    io = random.choice(b['io'])
                    inputs_list.append(torch.tensor(io[0], dtype=torch.float32))
                    targets_list.append(torch.tensor(io[1], dtype=torch.float32))
                inputs = torch.stack(inputs_list).to(device)
                targets = torch.stack(targets_list).to(device)
                pred_outputs = proxy(soft_exprs, inputs)
                loss_exec = criterion_exec(pred_outputs, targets)
                loss = loss_ce + config.get('lambda_def', 0.0) * loss_exec
            else:
                loss = loss_ce

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Eval on test
        model.eval()
        correct = 0
        with torch.no_grad():
            for d in test_data:
                desc = d['desc'].unsqueeze(0).to(device)
                pred_logits = model(desc, target_len=5)
                pred_ids = torch.argmax(pred_logits, dim=-1).squeeze(0).cpu()
                expr_tokens = [idx_to_token[str(idx.item())] for idx in pred_ids if idx.item() >= 4]
                io = d['io'][0]
                res = evaluate_rpn(expr_tokens, {'a': io[0][0], 'b': io[0][1], 'c': io[0][2]})
                if res is not None and abs(res - io[1]) < 1e-3:
                    correct += 1
        acc = correct / len(test_data)
        history.append(acc)
        print(f"Epoch {epoch+1}, Loss: {total_loss/(len(train_data)/batch_size):.4f}, Test Acc: {acc:.4f}")

    return history

def main():
    with open("diff_exec_feedback_code_gen/vocab.json", "r") as f:
        vocab = json.load(f)
    with open("diff_exec_feedback_code_gen/train.json", "r") as f:
        train_raw = json.load(f)
    with open("diff_exec_feedback_code_gen/test.json", "r") as f:
        test_raw = json.load(f)

    idx_to_token = {str(v): k for k, v in vocab.items()}

    def preprocess(data):
        processed = []
        for d in data:
            desc = d['description'].replace("(", " ").replace(")", " ").replace(",", " ").split()
            desc_ids = [vocab.get(t, vocab['<UNK>']) for t in desc]
            expr_ids = [vocab.get(t, vocab['<UNK>']) for t in d['expression']]
            io_pairs = [([io['inputs']['a'], io['inputs']['b'], io['inputs']['c']], io['output']) for io in d['io_pairs']]
            processed.append({'desc': torch.tensor(desc_ids), 'expr': torch.tensor(expr_ids), 'io': io_pairs})
        return processed

    train_all = preprocess(train_raw)
    test_data = preprocess(test_raw)

    train_subset = train_all[:2000]
    val_subset = train_all[2000:2500]

    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proxy = ProxyExecutor(vocab_size, 64, 256).to(device)
    proxy.load_state_dict(torch.load("diff_exec_feedback_code_gen/executor.pth", map_location=device))
    proxy.eval()

    print("Tuning Baseline...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda t: objective(t, 'baseline', train_subset, val_subset, vocab, vocab_size, proxy, idx_to_token), n_trials=5)

    print("Tuning DEF...")
    study_def = optuna.create_study(direction="maximize")
    study_def.optimize(lambda t: objective(t, 'def', train_subset, val_subset, vocab, vocab_size, proxy, idx_to_token), n_trials=5)

    print(f"Baseline Best Params: {study_baseline.best_params}, Val Acc: {study_baseline.best_value}")
    print(f"DEF Best Params: {study_def.best_params}, Val Acc: {study_def.best_value}")

    print("\nTraining Final Baseline...")
    history_baseline = train_final('baseline', study_baseline.best_params, train_all, test_data, vocab, vocab_size, proxy, idx_to_token)

    print("\nTraining Final DEF...")
    history_def = train_final('def', study_def.best_params, train_all, test_data, vocab, vocab_size, proxy, idx_to_token)

    plt.figure(figsize=(10, 6))
    plt.plot(history_baseline, label='Baseline (CE)')
    plt.plot(history_def, label='Proposed (CE + DEF)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (Functional)')
    plt.title('Code Generation: Baseline vs Differentiable Execution Feedback')
    plt.legend()
    plt.savefig('diff_exec_feedback_code_gen/results.png')

    results = {
        'baseline_best_val_acc': study_baseline.best_value,
        'def_best_val_acc': study_def.best_value,
        'baseline_final_test_acc': history_baseline[-1],
        'def_final_test_acc': history_def[-1]
    }
    with open('diff_exec_feedback_code_gen/results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
