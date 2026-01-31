import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OptiReLU(nn.Module):
    def __init__(self, input_dim, output_dim, lambda_reg=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_reg = lambda_reg

        # Neurons are stored as weights (K, input_dim) and biases (K,)
        self.hidden_weights = torch.empty((0, input_dim))
        self.hidden_biases = torch.empty((0,))

        # Output weights (output_dim, K)
        self.output_layer = nn.Linear(0, output_dim, bias=True)

    def forward(self, x):
        if self.hidden_weights.shape[0] == 0:
            # Fallback to bias only if no neurons
            return torch.zeros((x.shape[0], self.output_dim), device=x.device) + self.output_layer.bias

        # x: (B, input_dim)
        # hidden_weights: (K, input_dim)
        h = F.relu(F.linear(x, self.hidden_weights.to(x.device), self.hidden_biases.to(x.device)))
        return self.output_layer(h)

    def find_best_neuron(self, x, residuals, n_restarts=5, n_iters=100):
        """
        Find (w, b) that maximizes || sum_i grad_i * relu(w^T x_i + b) ||_2
        subject to ||(w, b)||_2 = 1.
        residuals: (B, output_dim)
        """
        device = x.device
        B, D = x.shape
        _, O = residuals.shape

        best_val = -float('inf')
        best_w = None
        best_b = None

        # Normalize x for better optimization
        # x_aug = [x, 1]
        x_aug = torch.cat([x, torch.ones((B, 1), device=device)], dim=1) # (B, D+1)

        for _ in range(n_restarts):
            # Random initialization on the unit sphere
            wb = torch.randn(D + 1, device=device, requires_grad=True)
            with torch.no_grad():
                wb /= torch.norm(wb)

            optimizer = torch.optim.Adam([wb], lr=0.1)

            for _ in range(n_iters):
                optimizer.zero_grad()
                wb_norm = wb / torch.norm(wb)

                # Projections onto neurons
                proj = F.relu(x_aug @ wb_norm) # (B,)

                # Correlation with residuals for each output dimension
                # corr_k = sum_i residuals_ik * proj_i
                corrs = torch.matmul(residuals.t(), proj) # (O,)

                # Objective: maximize L2 norm of correlations (or some other metric for multiclass)
                # We want to find a neuron that can help reduce loss across any/all classes.
                loss = -torch.norm(corrs)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                wb_norm = wb / torch.norm(wb)
                proj = F.relu(x_aug @ wb_norm)
                corrs = torch.matmul(residuals.t(), proj)
                val = torch.norm(corrs).item()
                if val > best_val:
                    best_val = val
                    best_w = wb_norm[:D].clone()
                    best_b = wb_norm[D].clone()

        return best_w, best_b

    def add_neuron(self, w, b):
        device = self.output_layer.weight.device
        self.hidden_weights = torch.cat([self.hidden_weights, w.unsqueeze(0).cpu()], dim=0)
        self.hidden_biases = torch.cat([self.hidden_biases, b.unsqueeze(0).cpu()], dim=0)

        # Update output layer
        K_old = self.output_layer.in_features
        K_new = K_old + 1
        new_output_layer = nn.Linear(K_new, self.output_dim, bias=True).to(device)

        with torch.no_grad():
            if K_old > 0:
                new_output_layer.weight[:, :K_old] = self.output_layer.weight
            new_output_layer.weight[:, K_old:] = 0 # Initialize new neuron weight to 0
            new_output_layer.bias.copy_(self.output_layer.bias)

        self.output_layer = new_output_layer

    def optimize_output_weights(self, x, y, n_iters=200):
        """
        Solve min Loss(f(x), y) + lambda * sum_j ||a_j||_2
        using proximal gradient descent or Adam.
        """
        if self.hidden_weights.shape[0] == 0:
            return

        device = x.device
        optimizer = torch.optim.Adam(self.output_layer.parameters(), lr=0.01)

        for _ in range(n_iters):
            optimizer.zero_grad()
            logits = self.forward(x)
            loss_data = F.cross_entropy(logits, y)

            # Group Lasso penalty
            # output_layer.weight is (O, K). We want norm over O for each K.
            reg = torch.norm(self.output_layer.weight, p=2, dim=0).sum()

            total_loss = loss_data + self.lambda_reg * reg
            total_loss.backward()
            optimizer.step()

            # Optional: proximal step for exact zeros
            with torch.no_grad():
                # Block soft-thresholding
                # a_j = self.output_layer.weight[:, j]
                # weight is (O, K)
                norms = torch.norm(self.output_layer.weight, p=2, dim=0) # (K,)
                # We should use a smaller learning rate for the proximal step or use a proper proximal grad
                # For simplicity here, just Adam + some pruning later
                pass

    def prune_neurons(self, threshold=1e-4):
        with torch.no_grad():
            norms = torch.norm(self.output_layer.weight, p=2, dim=0)
            keep_idx = norms > threshold

            if keep_idx.sum() == 0:
                # Keep at least the bias? No, let's keep neurons if they were useful
                return

            self.hidden_weights = self.hidden_weights[keep_idx.cpu()]
            self.hidden_biases = self.hidden_biases[keep_idx.cpu()]

            K_new = keep_idx.sum().item()
            new_output_layer = nn.Linear(K_new, self.output_dim, bias=True).to(self.output_layer.weight.device)
            new_output_layer.weight.copy_(self.output_layer.weight[:, keep_idx])
            new_output_layer.bias.copy_(self.output_layer.bias)
            self.output_layer = new_output_layer

def train_optirelu(model, x, y, max_neurons=50, lambda_reg=1e-3):
    model.lambda_reg = lambda_reg
    for i in range(max_neurons):
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            # One-hot y
            y_onehot = F.one_hot(y, num_classes=model.output_dim).float()
            # Gradient of cross-entropy w.r.t. logits is (probs - y_onehot)
            residuals = y_onehot - probs # (B, O) We want to maximize correlation with -grad, so (y - p)

        w, b = model.find_best_neuron(x, residuals)
        model.add_neuron(w, b)

        model.train()
        model.optimize_output_weights(x, y)

        if (i+1) % 5 == 0:
            model.prune_neurons()
            print(f"Iter {i+1}, Neurons: {model.hidden_weights.shape[0]}")

    model.prune_neurons(threshold=1e-5)
    return model
