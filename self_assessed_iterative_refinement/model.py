import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
import matplotlib.pyplot as plt
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

class IterativeRefinementModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_steps, use_loss_pred=False):
        super().__init__()
        self.num_steps = num_steps
        self.use_loss_pred = use_loss_pred

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.head = nn.Linear(hidden_dim, num_classes)
        if use_loss_pred:
            self.loss_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, x):
        h = self.encoder(x)
        logits_list = []
        loss_preds_list = []

        for _ in range(self.num_steps):
            # Refine h based on current h and original input x
            h = h + self.refiner(torch.cat([h, x], dim=-1))

            logits = self.head(h)
            logits_list.append(logits)

            if self.use_loss_pred:
                loss_pred = self.loss_predictor(h)
                loss_preds_list.append(loss_pred)

        return logits_list, loss_preds_list

def train_and_evaluate(config):
    X_train, y_train, X_test, y_test = get_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=config['batch_size'], shuffle=True)

    model = IterativeRefinementModel(
        input_dim=40,
        hidden_dim=config['hidden_dim'],
        num_classes=10,
        num_steps=config['num_steps'],
        use_loss_pred=config['use_loss_pred']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits_list, loss_preds_list = model(batch_x)

            total_loss = 0
            for t in range(len(logits_list)):
                ce_loss = F.cross_entropy(logits_list[t], batch_y)
                total_loss += ce_loss

                if config['use_loss_pred']:
                    # Target for loss prediction is the actual CE loss
                    # We detach the target to avoid backpropping through the CE loss for the predictor's head
                    target_loss = ce_loss.detach().view(-1, 1).expand(batch_x.size(0), 1)
                    # Actually, ce_loss is a scalar (mean over batch).
                    # If we want per-sample loss prediction, we should use reduction='none'
                    ce_loss_none = F.cross_entropy(logits_list[t], batch_y, reduction='none').view(-1, 1)
                    mse_loss = F.mse_loss(loss_preds_list[t], ce_loss_none.detach())
                    total_loss += config['lambda_loss'] * mse_loss

            total_loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits_list, _ = model(X_test)
        final_logits = test_logits_list[-1]
        preds = final_logits.argmax(dim=-1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy

if __name__ == "__main__":
    # This is a placeholder for the actual experiment script
    pass
