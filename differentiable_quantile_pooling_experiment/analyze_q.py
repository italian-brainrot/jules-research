import torch
import torch.nn as nn
import torch.optim as optim
from models import Net
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import numpy as np
import matplotlib.pyplot as plt

def analyze():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_lr = 0.008674

    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)

    model = Net('quantile').to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    criterion = nn.CrossEntropyLoss()

    print("Training quantile model to analyze learned q...")
    for epoch in range(15):
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/15 done.")

    q1 = torch.sigmoid(model.pool1.q_raw).detach().cpu().numpy()
    q2 = torch.sigmoid(model.pool2.q_raw).detach().cpu().numpy()

    print("\nLearned q values for Pool1 (first 10 channels):", q1[:10])
    print("Learned q values for Pool2 (first 10 channels):", q2[:10])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(q1, bins=20)
    plt.title("Pool1 Learned q")
    plt.xlabel("q")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(q2, bins=20)
    plt.title("Pool2 Learned q")
    plt.xlabel("q")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("differentiable_quantile_pooling_experiment/learned_q.png")
    print("\nHistograms saved to differentiable_quantile_pooling_experiment/learned_q.png")

    # Also save the distribution to results.md
    with open("differentiable_quantile_pooling_experiment/results.md", "a") as f:
        f.write("\n## Analysis of Learned Quantiles\n\n")
        f.write(f"Average q in Pool1: {np.mean(q1):.4f}\n")
        f.write(f"Average q in Pool2: {np.mean(q2):.4f}\n\n")
        f.write("![Learned q Histograms](learned_q.png)\n")

if __name__ == "__main__":
    analyze()
