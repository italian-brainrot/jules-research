import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import matplotlib.pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_data():
    args = get_dataset_args()
    args.num_samples = 5000
    data = make_dataset(args)
    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return x_train, y_train, x_test, y_test

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def run():
    set_seed()
    x_train, y_train, x_test, y_test = get_data()
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    gsnr_history = {name: [] for name, _ in model.named_parameters() if 'weight' in name}

    epochs = 100
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # Estimate GSNR from Adam's buffers
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        state = optimizer.state[p]
                        if 'exp_avg' in state and 'exp_avg_sq' in state:
                            m = state['exp_avg']
                            v = state['exp_avg_sq']
                            # Simple GSNR estimate
                            gsnr = (m.pow(2) / (v + 1e-8)).mean().item()
                            gsnr_history[name].append(gsnr)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} done")

    plt.figure(figsize=(10, 6))
    for name, gsnrs in gsnr_history.items():
        plt.plot(gsnrs, label=name)
    plt.title('GSNR Estimation during training')
    plt.xlabel('Steps')
    plt.ylabel('GSNR')
    plt.legend()
    plt.savefig('gsnr_adaptive_dropout_experiment/gsnr_check.png')
    print("GSNR check plot saved.")

if __name__ == '__main__':
    run()
