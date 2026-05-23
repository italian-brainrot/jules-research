import torch
import torch.nn as nn
import torch.fft

class DVMD(nn.Module):
    def __init__(self, n_modes=3, alpha=2000, n_iter=20, tol=1e-6):
        super().__init__()
        self.K = n_modes
        self.alpha = nn.Parameter(torch.full((n_modes,), float(alpha)))
        self.n_iter = n_iter
        self.tol = tol

    def forward(self, x):
        # x shape: (batch, seq_len)
        batch_size, T = x.shape

        # Spectral domain
        f = torch.fft.fft(x)

        # Positive frequencies only (analytic signal)
        half_T = T // 2 + 1
        f_pos = f[:, :half_T].clone()
        if T % 2 == 0:
            f_pos[:, 1:T//2] *= 2.0
        else:
            f_pos[:, 1:] *= 2.0

        freqs = torch.linspace(0, 0.5, half_T, device=x.device)

        # Initialize
        u_hat_list = [torch.zeros((batch_size, half_T), dtype=torch.complex64, device=x.device) for _ in range(self.K)]
        # Center frequencies initialized uniformly in [0, 0.5]
        omega = torch.linspace(0, 0.5, self.K + 2, device=x.device)[1:-1].repeat(batch_size, 1)
        lambd_hat = torch.zeros((batch_size, half_T), dtype=torch.complex64, device=x.device)

        for i in range(self.n_iter):
            # sum_all_u = sum(u_hat_list)
            sum_all_u = torch.stack(u_hat_list, dim=1).sum(dim=1)

            new_omegas = []
            for k in range(self.K):
                # Update u_hat[k]
                sum_u_neq_k = sum_all_u - u_hat_list[k]

                # Residual
                res = f_pos - sum_u_neq_k + lambd_hat / 2.0

                # Wiener Filter Kernel
                dist = (freqs.unsqueeze(0) - omega[:, k].unsqueeze(1))**2
                kernel = 1.0 / (1.0 + 2.0 * self.alpha[k] * dist)

                new_u_hat_k = res * kernel

                # Update sum_all_u for the next mode in the same iteration (Gauss-Seidel)
                sum_all_u = sum_all_u - u_hat_list[k] + new_u_hat_k
                u_hat_list[k] = new_u_hat_k

                # Update omega_k (center frequency)
                u_sq = torch.abs(u_hat_list[k])**2
                u_sum = torch.sum(u_sq, dim=1) + 1e-10
                new_omega_k = torch.sum(freqs.unsqueeze(0) * u_sq, dim=1) / u_sum
                new_omegas.append(new_omega_k)

            omega = torch.stack(new_omegas, dim=1)

            # Update lambd_hat (dual ascent)
            res_total = f_pos - sum_all_u
            lambd_hat = lambd_hat + 0.1 * res_total

        # Features: mode energies and center frequencies
        u_hat_tensor = torch.stack(u_hat_list, dim=1)
        mode_energies = torch.sum(torch.abs(u_hat_tensor)**2, dim=2)

        # Sort by omega to have a consistent order for the MLP
        # This is important because the order of modes might flip during training
        idx = torch.argsort(omega, dim=1)
        mode_energies = torch.gather(mode_energies, 1, idx)
        omega = torch.gather(omega, 1, idx)

        return mode_energies, omega

if __name__ == "__main__":
    # Simple test
    vmd = DVMD(n_modes=2, n_iter=20)
    t = torch.linspace(0, 1, 40)
    # x = sin(2*pi*5*t) + 0.5*sin(2*pi*15*t)
    # 5 Hz is 5/40 = 0.125 normalized freq
    # 15 Hz is 15/40 = 0.375 normalized freq
    x = torch.sin(2 * 3.14159 * 5 * t) + 0.5 * torch.sin(2 * 3.14159 * 15 * t)
    x = x.unsqueeze(0)

    energies, omegas = vmd(x)
    print("Energies:", energies)
    print("Omegas:", omegas)

    # Check gradients
    x.requires_grad = True
    energies, omegas = vmd(x)
    loss = energies.sum() + omegas.sum()
    loss.backward()
    print("x grad norm:", x.grad.norm().item())
    print("alpha grad:", vmd.alpha.grad)
