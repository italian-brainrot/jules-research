import matplotlib.pyplot as plt
import ast
import os

def read_results(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        res = {}
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                res[key.strip()] = val.strip()
    return res

baseline_res = read_results('differentiable_approximate_entropy/results_baseline.txt')
dsampen_res = read_results('differentiable_approximate_entropy/results_dsampen.txt')

baseline_hist = ast.literal_eval(baseline_res['History'])
dsampen_hist = ast.literal_eval(dsampen_res['History'])

plt.figure(figsize=(10, 6))
plt.plot(baseline_hist, label=f"Baseline (Mean Acc: {float(baseline_res['Mean']):.4f})")
plt.plot(dsampen_hist, label=f"DSampEn (Mean Acc: {float(dsampen_res['Mean']):.4f})")
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('MNIST-1D: Baseline vs DSampEn Augmented MLP')
plt.legend()
plt.grid(True)
plt.savefig('differentiable_approximate_entropy/comparison.png')
print("Comparison plot saved.")
