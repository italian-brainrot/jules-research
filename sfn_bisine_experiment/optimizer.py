import torch
import torch.nn as nn
import numpy as np

class SFNOptimizer:
    def __init__(self, model, lr=1.0, eps=1e-4, damping=1e-3):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.damping = damping
        self.C = model.num_classes
        self.P_c = model.params_per_class
        self.P = self.C * self.P_c

    def step(self, x, y):
        N = x.shape[0]
        device = x.device
        z = self.model(x)
        p = torch.softmax(z, dim=1)
        y_oh = torch.zeros_like(z)
        y_oh.scatter_(1, y.unsqueeze(1), 1.0)
        loss = -torch.mean(torch.sum(y_oh * torch.log(p + 1e-10), dim=1))
        G_model, H_model = self.model.compute_grad_and_hessian(x)
        dL_dz = (p - y_oh) / N
        grad = torch.sum(dL_dz.unsqueeze(2) * G_model, dim=0)
        grad_flat = grad.view(-1)
        full_H = torch.zeros(self.P, self.P, device=device)
        for c in range(self.C):
            gnc = G_model[:, c, :]
            w1 = (p[:, c] - p[:, c]**2) / N
            w2 = (p[:, c] - y_oh[:, c]) / N
            H_cc = (gnc.t() * w1) @ gnc
            H_model_c_sum = torch.sum(w2.view(N, 1, 1) * H_model[:, c, :, :], dim=0)
            H_cc += H_model_c_sum
            full_H[c*self.P_c : (c+1)*self.P_c, c*self.P_c : (c+1)*self.P_c] = H_cc
            for d in range(c + 1, self.C):
                gnd = G_model[:, d, :]
                w_cd = (-p[:, c] * p[:, d]) / N
                H_cd = (gnc.t() * w_cd) @ gnd
                full_H[c*self.P_c : (c+1)*self.P_c, d*self.P_c : (d+1)*self.P_c] = H_cd
                full_H[d*self.P_c : (d+1)*self.P_c, c*self.P_c : (c+1)*self.P_c] = H_cd.t()
        L, V = torch.linalg.eigh(full_H)
        L_abs = torch.abs(L)
        L_inv = 1.0 / (L_abs + self.damping)
        delta_theta = - (V @ torch.diag(L_inv) @ V.t() @ grad_flat)
        current_params = self.model.get_flat_params()
        alpha = self.lr
        c_ls = 1e-4
        tau = 0.5
        best_alpha = 0.0
        orig_loss = loss.item()
        for _ in range(10):
            new_params = current_params + alpha * delta_theta
            self.model.set_flat_params(new_params)
            with torch.no_grad():
                z_new = self.model(x)
                p_new = torch.softmax(z_new, dim=1)
                new_loss = -torch.mean(torch.sum(y_oh * torch.log(p_new + 1e-10), dim=1)).item()
            if new_loss < orig_loss + c_ls * alpha * torch.dot(grad_flat, delta_theta):
                best_alpha = alpha
                break
            alpha *= tau
        if best_alpha == 0.0:
            self.model.set_flat_params(current_params)
        else:
            self.model.set_flat_params(current_params + best_alpha * delta_theta)
        return orig_loss, L
