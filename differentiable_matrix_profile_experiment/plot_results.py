import matplotlib.pyplot as plt
import numpy as np

# Data from results.txt
models = ['Baseline MLP', 'MP-Augmented MLP']
accuracies = [0.7875, 0.7730]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green'])
plt.ylabel('Test Accuracy')
plt.title('MNIST-1D Classification Performance')
plt.ylim(0, 1.0)

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

plt.savefig('differentiable_matrix_profile_experiment/comparison.png')
print("Comparison plot saved as comparison.png")
