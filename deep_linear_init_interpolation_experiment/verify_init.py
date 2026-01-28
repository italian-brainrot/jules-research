import torch
import numpy as np
from deep_linear_init_interpolation_experiment.compare import LongThinNet, compute_ls_solution, ls_init
from mnist1d.data import make_dataset, get_dataset_args

def test_verification():
    defaults = get_dataset_args()
    defaults.num_samples = 2000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()

    W_ls, b_ls = compute_ls_solution(X_train, y_train)

    model = LongThinNet(input_size=40, hidden_size=10, output_size=10, num_layers=16, alpha=0.0)
    ls_init(model, W_ls, b_ls)

    model.eval()
    with torch.no_grad():
        X_sample = X_train[:10]
        y_pred_model = model(X_sample)
        y_pred_ls = X_sample @ W_ls + b_ls

        diff = torch.abs(y_pred_model - y_pred_ls).max()
        print(f"Max difference between model and LS: {diff.item():.2e}")

        if diff < 1e-4:
            print("Verification SUCCESSFUL")
        else:
            print("Verification FAILED")

if __name__ == "__main__":
    test_verification()
