import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import matplotlib.pyplot as plt
import time

def to_one_hot(y, num_classes=10):
    oh = torch.zeros(len(y), num_classes, device=y.device)
    oh.scatter_(1, y.unsqueeze(1), 1)
    return oh

def solve_least_squares(X, Y):
    # Solves W [X, 1] = Y => W = Y [X, 1]^\dagger
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    # torch.linalg.lstsq(A, B) solves A X = B.
    # Here X_aug W^T = Y => A=X_aug, B=Y, X_sol = W^T.
    sol = torch.linalg.lstsq(X_aug, Y).solution
    W_aug = sol.t()
    return W_aug[:, :-1], W_aug[:, -1]

class PrependedGreedyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation='leaky_relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        if activation == 'leaky_relu':
            self.sigma = nn.LeakyReLU(0.1)
            self.sigma_inv = lambda y: torch.where(y > 0, y, y / 0.1)
        elif activation == 'identity':
            self.sigma = nn.Identity()
            self.sigma_inv = lambda y: y
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):
        if len(self.layers) == 0:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)

        res = x
        # L_k is at the end of ModuleList. Forward: L_0(sigma(L_1(...sigma(L_k(x))...)))
        for i in range(len(self.layers) - 1, 0, -1):
            res = self.layers[i](res)
            res = self.sigma(res)
        res = self.layers[0](res)
        return res

    def add_layer(self, W, b):
        layer = nn.Linear(W.shape[1], W.shape[0])
        layer.weight.data.copy_(W)
        layer.bias.data.copy_(b)
        self.layers.append(layer)

def train_pgl(X, Y_oh, y, X_test, y_test, num_layers=5, z_steps=1000, z_lr=0.01, lambda_lin=0.0):
    device = X.device
    input_dim = X.shape[1]
    output_dim = Y_oh.shape[1]
    hidden_dim = input_dim

    net = PrependedGreedyNet(input_dim, output_dim, hidden_dim).to(device)

    # Step 0: Fit initial linear layer
    W0, b0 = solve_least_squares(X, Y_oh)
    net.add_layer(W0, b0)

    accuracies = []
    losses = []

    def eval_model(model):
        model.eval()
        with torch.no_grad():
            out = model(X_test)
            preds = out.argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

            out_train = model(X)
            loss = nn.MSELoss()(out_train, Y_oh).item()
        return acc, loss

    acc, loss = eval_model(net)
    accuracies.append(acc)
    losses.append(loss)
    print(f"Layer 0 Acc: {acc:.4f}, Loss: {loss:.6f}")

    for k in range(1, num_layers):
        # Optimize Z to minimize loss
        Z = X.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([Z], lr=z_lr)
        criterion = nn.MSELoss()

        for _ in range(z_steps):
            optimizer.zero_grad()
            out = net(Z)
            loss_val = criterion(out, Y_oh)
            if lambda_lin > 0:
                loss_val += lambda_lin * torch.mean((Z - X)**2)
            loss_val.backward()
            optimizer.step()

        # Fit new layer L_k s.t. sigma(L_k(X)) approx Z
        target = net.sigma_inv(Z.detach())
        Wk, bk = solve_least_squares(X, target)

        net.add_layer(Wk, bk)

        acc, loss = eval_model(net)
        accuracies.append(acc)
        losses.append(loss)
        print(f"Layer {k} Acc: {acc:.4f}, Loss: {loss:.6f}")

    return accuracies, losses

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_mlp(X, y, Y_oh, X_test, y_test, num_layers, epochs=500, lr=0.001):
    input_dim = X.shape[1]
    output_dim = 10
    hidden_dim = input_dim
    model = MLP(input_dim, output_dim, hidden_dim, num_layers).to(X.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dl = TensorDataLoader((X, y), batch_size=256, shuffle=True)

    for epoch in range(epochs):
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(X_test)
        acc = (out.argmax(dim=1) == y_test).float().mean().item()

        out_train = model(X)
        loss_mse = nn.MSELoss()(to_one_hot(out_train.argmax(dim=1)), Y_oh).item() # Not perfect but for comparison

    return acc

def main():
    print("Setting up dataset...")
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(data['x'], dtype=torch.float32).to(device)
    y_train = torch.tensor(data['y'], dtype=torch.long).to(device)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32).to(device)
    y_test = torch.tensor(data['y_test'], dtype=torch.long).to(device)

    Y_train_oh = to_one_hot(y_train)
    Y_test_oh = to_one_hot(y_test)

    num_layers = 6

    print("\nTraining Prepended Greedy (Basic)...")
    pgl_basic_acc, pgl_basic_loss = train_pgl(X_train, Y_train_oh, y_train, X_test, y_test, num_layers=num_layers, lambda_lin=0.0)

    print("\nTraining Prepended Greedy (Linear Constrained lambda=0.1)...")
    pgl_lin_acc, pgl_lin_loss = train_pgl(X_train, Y_train_oh, y_train, X_test, y_test, num_layers=num_layers, lambda_lin=0.1)

    print("\nTraining Prepended Greedy (Linear Constrained lambda=1.0)...")
    pgl_lin2_acc, pgl_lin2_loss = train_pgl(X_train, Y_train_oh, y_train, X_test, y_test, num_layers=num_layers, lambda_lin=1.0)

    print("\nTraining MLP Baselines...")
    mlp_accs = []
    for l in range(1, num_layers + 1):
        acc = train_mlp(X_train, y_train, Y_train_oh, X_test, y_test, num_layers=l)
        mlp_accs.append(acc)
        print(f"MLP with {l} layers Acc: {acc:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    layers = np.arange(num_layers)
    plt.plot(layers, pgl_basic_acc, marker='o', label='PGL Basic')
    plt.plot(layers, pgl_lin_acc, marker='s', label='PGL Linear (lambda=0.1)')
    plt.plot(layers, pgl_lin2_acc, marker='^', label='PGL Linear (lambda=1.0)')
    plt.plot(layers, mlp_accs, marker='x', label='MLP (Adam)')
    plt.xlabel('Number of Layers')
    plt.ylabel('Test Accuracy')
    plt.title('Prepended Greedy Layer-wise Learning on MNIST1D')
    plt.legend()
    plt.grid(True)
    plt.savefig('prepended_greedy_layerwise_experiment/accuracy_comparison.png')

    plt.figure(figsize=(10, 6))
    plt.plot(layers, pgl_basic_loss, marker='o', label='PGL Basic')
    plt.plot(layers, pgl_lin_acc, marker='s', label='PGL Linear (lambda=0.1)')
    plt.plot(layers, pgl_lin2_acc, marker='^', label='PGL Linear (lambda=1.0)')
    plt.xlabel('Number of Layers')
    plt.ylabel('Training MSE Loss')
    plt.title('Prepended Greedy Layer-wise Learning - Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('prepended_greedy_layerwise_experiment/loss_comparison.png')

    print("\nResults:")
    print(f"PGL Basic Acc: {pgl_basic_acc}")
    print(f"PGL Lin (0.1) Acc: {pgl_lin_acc}")
    print(f"PGL Lin (1.0) Acc: {pgl_lin2_acc}")
    print(f"MLP Acc: {mlp_accs}")

if __name__ == "__main__":
    main()
