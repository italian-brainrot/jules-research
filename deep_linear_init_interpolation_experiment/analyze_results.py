import pickle
import os

with open('deep_linear_init_interpolation_experiment/results.pkl', 'rb') as f:
    results = pickle.load(f)

for config, history in results.items():
    best_val_acc = max(history['val_acc'])
    final_val_acc = history['val_acc'][-1]
    print(f"{config}: Best Val Acc = {best_val_acc:.4f}, Final Val Acc = {final_val_acc:.4f}")
