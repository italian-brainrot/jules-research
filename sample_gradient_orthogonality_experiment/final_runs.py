import torch
from torch import nn
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from sgo_lib import MLP, get_sgo_penalty
import matplotlib.pyplot as plt
import json

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(mode, lr, lambda_reg, epochs=100):
    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'test_acc': [], 'train_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for x, y in dl_train:
            optimizer.zero_grad()
            logits = model(x)
            loss_ce = nn.functional.cross_entropy(logits, y)

            loss = loss_ce
            if mode == 'SGO':
                params = dict(model.named_parameters())
                penalty = get_sgo_penalty(params, model, x, y, class_aware=False)
                loss = loss + lambda_reg * penalty
            elif mode == 'CSGO':
                params = dict(model.named_parameters())
                penalty = get_sgo_penalty(params, model, x, y, class_aware=True)
                loss = loss + lambda_reg * penalty

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        history['train_loss'].append(epoch_loss / len(dl_train))
        history['train_acc'].append(correct / total)
        history['test_acc'].append(test_acc)

    return history

best_params = {
    'Baseline': {'lr': 0.0015625243614928487, 'lambda_reg': 0.0},
    'SGO': {'lr': 0.002662364017371652, 'lambda_reg': 0.06225339993327537},
    'CSGO': {'lr': 0.0008594901156371331, 'lambda_reg': 0.0021756203998476013}
}

histories = {}
for mode, params in best_params.items():
    print(f"Final run for {mode}...")
    histories[mode] = train_model(mode, params['lr'], params['lambda_reg'], epochs=100)

# Save histories
with open('histories.json', 'w') as f:
    json.dump(histories, f)

# Print final results
print("\nFinal Results (at 100 epochs):")
for mode in best_params.keys():
    h = histories[mode]
    print(f"{mode}:")
    print(f"  Train Loss: {h['train_loss'][-1]:.4f}")
    print(f"  Train Acc: {h['train_acc'][-1]:.4f}")
    print(f"  Test Acc: {h['test_acc'][-1]:.4f}")
    print(f"  Best Test Acc: {max(h['test_acc']):.4f}")

# Plot
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
for mode in best_params.keys():
    plt.plot(histories[mode]['train_loss'], label=mode)
plt.title('Train Loss')
plt.legend()

plt.subplot(1, 3, 2)
for mode in best_params.keys():
    plt.plot(histories[mode]['train_acc'], label=mode)
plt.title('Train Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
for mode in best_params.keys():
    plt.plot(histories[mode]['test_acc'], label=mode)
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('final_comparison.png')
