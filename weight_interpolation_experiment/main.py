
import torch
import torch.nn as nn
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import collections

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- 1. Define the Model ---
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    set_seed(42)

    # --- 2. Load Data ---
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.int64)
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    # --- 3. Training Setup ---
    input_size = X_train.shape[1]
    hidden_size = 256
    output_size = y_train.max().item() + 1

    model = MLP(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Training Loop & Snapshot Saving ---
    epochs = 50
    snapshot_epochs = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 49]
    snapshots = collections.OrderedDict()

    for epoch in range(epochs):
        if epoch in snapshot_epochs:
            print(f"Saving snapshot at epoch {epoch}...")
            snapshots[epoch] = collections.OrderedDict({k: v.clone() for k, v in model.state_dict().items()})

        for i, (inputs, targets) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # --- 5. Save Snapshots ---
    torch.save(snapshots, 'weight_interpolation_experiment/snapshots.pt')
    print("Training complete and snapshots saved to weight_interpolation_experiment/snapshots.pt")

if __name__ == "__main__":
    main()
