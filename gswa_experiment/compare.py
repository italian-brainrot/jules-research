import torch
import matplotlib.pyplot as plt
import numpy as np
import mnist1d
import os
from gswa_experiment.model import MLP
from gswa_experiment.trainer import train, evaluate
from light_dataloader import TensorDataLoader

def get_data():
    args = mnist1d.get_dataset_args()
    data = mnist1d.get_dataset(args, path='gswa_experiment/mnist1d_data.pkl')
    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    n_train = int(0.8 * len(x_train))
    train_ds = (x_train[:n_train], y_train[:n_train])
    val_ds = (x_train[n_train:], y_train[n_train:])

    return train_ds, val_ds, (x_test, y_test)

def run_experiment(best_lrs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, val_ds, test_ds = get_data()

    train_loader = TensorDataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = TensorDataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = TensorDataLoader(test_ds, batch_size=128, shuffle=False)

    seeds = [42, 43, 44, 45, 46]
    results = {}

    for mode in ['Adam', 'SWA', 'GSWA']:
        print(f"Running final evaluation for {mode}...")
        mode_accs = []
        mode_histories = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = MLP().to(device)
            lr = best_lrs[mode]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_model, history = train(model, train_loader, val_loader, optimizer, device, epochs=50, mode=mode)
            acc = evaluate(best_model, test_loader, device)
            mode_accs.append(acc)
            mode_histories.append(history)
            print(f"  Seed {seed}: Acc = {acc:.4f}")

        results[mode] = {
            'accs': mode_accs,
            'histories': mode_histories,
            'mean_acc': np.mean(mode_accs),
            'std_acc': np.std(mode_accs)
        }

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for mode in ['Adam', 'SWA', 'GSWA']:
        # Average train loss over seeds
        losses = np.array([h['train_loss'] for h in results[mode]['histories']])
        plt.plot(np.mean(losses, axis=0), label=mode)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in ['Adam', 'SWA', 'GSWA']:
        accs = np.array([h['val_acc'] for h in results[mode]['histories']])
        plt.plot(np.mean(accs, axis=0), label=mode)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gswa_experiment/comparison.png')

    # Save results to text
    with open('gswa_experiment/results.txt', 'w') as f:
        for mode in ['Adam', 'SWA', 'GSWA']:
            f.write(f"{mode}:\n")
            f.write(f"  Mean Acc: {results[mode]['mean_acc']:.4f} +/- {results[mode]['std_acc']:.4f}\n")
            f.write(f"  Best LRs: {best_lrs[mode]}\n")
            f.write(f"  Accs: {results[mode]['accs']}\n\n")

if __name__ == "__main__":
    # These would normally come from tune.py.
    # For now, I will run tune.py first and then pass the results or just hardcode them if I run them together.
    # Since I'm supposed to run tune.py in step 4, I'll make compare.py read them or take them as input.
    import sys
    if len(sys.argv) > 1:
        import json
        best_lrs = json.loads(sys.argv[1])
    else:
        # Default fallback or manual entry if needed
        best_lrs = {'Adam': 0.001, 'SWA': 0.001, 'GSWA': 0.001}

    run_experiment(best_lrs)
