import torch
import matplotlib.pyplot as plt
from train import get_data, train_model
from model import SoftWalshNetwork, MLP
from light_dataloader import TensorDataLoader
import json
import os

X_train, y_train, X_test, y_test = get_data()
train_loader = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
test_loader = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

# Load best params from tuning
with open("soft_walsh_experiment/mlp_params.json", "r") as f:
    mlp_hparams = json.load(f)
with open("soft_walsh_experiment/swn_params.json", "r") as f:
    swn_hparams = json.load(f)

print(f"MLP Params: {mlp_hparams}")
print(f"SWN Params: {swn_hparams}")

print("Final training MLP...")
mlp = MLP(40, mlp_hparams['hidden_size'], 10, num_layers=mlp_hparams['num_layers'])
mlp_acc, mlp_history = train_model(mlp, train_loader, test_loader, epochs=50,
                                  lr=mlp_hparams['lr'], weight_decay=mlp_hparams['weight_decay'], verbose=True)

print("\nFinal training SWN...")
swn = SoftWalshNetwork(40, swn_hparams['nt'], 10, deep=swn_hparams['dp'], init_scale=swn_hparams['iscl'])
swn_acc, swn_history = train_model(swn, train_loader, test_loader, epochs=100,
                                  lr=swn_hparams['lr'], weight_decay=swn_hparams['wd'],
                                  sparsity_lambda=swn_hparams['sl'], verbose=True)

torch.save(swn.state_dict(), "soft_walsh_experiment/swn_weights.pt")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(mlp_history, label=f"MLP (Best: {mlp_acc:.4f})")
plt.plot(swn_history, label=f"SWN (Best: {swn_acc:.4f})")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Comparison: MLP vs Soft Walsh Network")
plt.legend()
plt.grid(True)
plt.savefig("soft_walsh_experiment/comparison_plot.png")

# Save summary
with open("soft_walsh_experiment/results.json", "w") as f:
    json.dump({
        "mlp_acc": mlp_acc,
        "swn_acc": swn_acc,
        "mlp_history": mlp_history,
        "swn_history": swn_history,
        "mlp_params": mlp_hparams,
        "swn_params": swn_hparams
    }, f, indent=4)

print(f"\nResults saved to soft_walsh_experiment/results.json")
print(f"Plot saved to soft_walsh_experiment/comparison_plot.png")
