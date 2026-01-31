import torch
from model import BisineNetwork
import torch.autograd.functional as F

def verify():
    input_dim = 5
    num_classes = 3
    num_units = 2
    batch_size = 4

    model = BisineNetwork(input_dim, num_classes, num_units)
    x = torch.randn(batch_size, input_dim)

    G_manual, H_manual = model.compute_grad_and_hessian(x)

    for c in range(num_classes):
        def func_class(params_c):
            z_c = torch.zeros(batch_size)
            for k in range(num_units):
                start = k * model.params_per_unit
                a = params_c[start]
                w1 = params_c[start + 1 : start + 1 + input_dim]
                b1 = params_c[start + 1 + input_dim]
                w2 = params_c[start + 2 + input_dim : start + 2 + 2 * input_dim]
                b2 = params_c[start + 2 + 2 * input_dim]
                u1 = torch.matmul(x, w1) + b1
                u2 = torch.matmul(x, w2) + b2
                z_c += a * torch.sin(u1) * torch.sin(u2)
            return z_c

        params_c = model.params[c]
        G_auto = F.jacobian(func_class, params_c)
        H_auto = torch.zeros(batch_size, model.params_per_class, model.params_per_class)
        for n in range(batch_size):
            def func_sample(params_c_sample, sample_idx=n):
                z = torch.zeros(batch_size)
                for k in range(num_units):
                    start = k * model.params_per_unit
                    a = params_c_sample[start]
                    w1 = params_c_sample[start + 1 : start + 1 + input_dim]
                    b1 = params_c_sample[start + 1 + input_dim]
                    w2 = params_c_sample[start + 2 + input_dim : start + 2 + 2 * input_dim]
                    b2 = params_c_sample[start + 2 + 2 * input_dim]
                    u1 = torch.matmul(x[sample_idx], w1) + b1
                    u2 = torch.matmul(x[sample_idx], w2) + b2
                    z[sample_idx] += a * torch.sin(u1) * torch.sin(u2)
                return z[sample_idx]
            H_auto[n] = F.hessian(func_sample, params_c)

        diff_G = torch.norm(G_manual[:, c, :] - G_auto)
        diff_H = torch.norm(H_manual[:, c, :, :] - H_auto)
        print(f"Class {c}: Grad Diff = {diff_G.item():.2e}, Hessian Diff = {diff_H.item():.2e}")
        if diff_G > 1e-4 or diff_H > 1e-4:
            return False
    return True

if __name__ == "__main__":
    if verify():
        print("Verification PASSED!")
    else:
        print("Verification FAILED!")
