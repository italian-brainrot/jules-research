import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q, alpha, iterations=20):
        K = x.shape[-1]
        m_min = x.min(dim=-1).values
        m_max = x.max(dim=-1).values
        m_min = m_min - 2.0 / alpha
        m_max = m_max + 2.0 / alpha
        for _ in range(iterations):
            m_mid = (m_min + m_max) / 2
            diff = alpha.unsqueeze(-1) * (m_mid.unsqueeze(-1) - x)
            vals = torch.sigmoid(diff).mean(dim=-1)
            too_high = vals > q
            m_max = torch.where(too_high, m_mid, m_max)
            m_min = torch.where(too_high, m_min, m_mid)
        m = (m_min + m_max) / 2
        ctx.save_for_backward(x, q, alpha, m)
        return m

    @staticmethod
    def backward(ctx, grad_output):
        x, q, alpha, m = ctx.saved_tensors
        K = x.shape[-1]
        diff = alpha.unsqueeze(-1) * (m.unsqueeze(-1) - x)
        s = torch.sigmoid(diff)
        sp = s * (1 - s)
        sum_sp = sp.sum(dim=-1) + 1e-10
        grad_x = grad_output.unsqueeze(-1) * (sp / sum_sp.unsqueeze(-1))
        grad_q = grad_output * (K / (alpha * sum_sp))
        grad_alpha = - grad_output * ((m.unsqueeze(-1) - x) * sp).sum(dim=-1) / (alpha * sum_sp)
        return grad_x, grad_q, grad_alpha, None

class ImplicitQuantilePooling1d(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, padding=0, initial_q=0.5, alpha=10.0, learn_q=True, learn_alpha=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.alpha_raw = nn.Parameter(torch.full((channels,), np.log(alpha)), requires_grad=learn_alpha)
        self.q_raw = nn.Parameter(torch.full((channels,), float(torch.logit(torch.tensor(initial_q)))), requires_grad=learn_q)

    def forward(self, x):
        B, C, L = x.shape
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode='replicate')
        x_unfolded = x.unfold(2, self.kernel_size, self.stride)
        q = torch.sigmoid(self.q_raw)
        alpha = torch.exp(self.alpha_raw)
        q_b = q.view(1, C, 1).expand(B, C, x_unfolded.shape[2])
        alpha_b = alpha.view(1, C, 1).expand(B, C, x_unfolded.shape[2])
        m = QuantileFunction.apply(x_unfolded, q_b, alpha_b)
        return m

class MedianPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode='replicate')
        x_unfolded = x.unfold(2, self.kernel_size, self.stride)
        return x_unfolded.median(dim=-1).values

class LpPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, p=2.0, learn_p=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.p_raw = nn.Parameter(torch.tensor(np.log(p)), requires_grad=learn_p)
    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode='replicate')
        x_unfolded = x.unfold(2, self.kernel_size, self.stride)
        p = torch.exp(self.p_raw)
        return (x_unfolded.abs().pow(p).mean(dim=-1) + 1e-10).pow(1.0/p)

class Net(nn.Module):
    def __init__(self, pool_type, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=5, padding=2)

        if pool_type == 'max':
            self.pool1 = nn.MaxPool1d(kernel_size=2)
            self.pool2 = nn.MaxPool1d(kernel_size=2)
        elif pool_type == 'avg':
            self.pool1 = nn.AvgPool1d(kernel_size=2)
            self.pool2 = nn.AvgPool1d(kernel_size=2)
        elif pool_type == 'median':
            self.pool1 = MedianPool1d(kernel_size=2)
            self.pool2 = MedianPool1d(kernel_size=2)
        elif pool_type == 'lp':
            self.pool1 = LpPool1d(kernel_size=2)
            self.pool2 = LpPool1d(kernel_size=2)
        elif pool_type == 'quantile':
            self.pool1 = ImplicitQuantilePooling1d(hidden_channels, kernel_size=2)
            self.pool2 = ImplicitQuantilePooling1d(hidden_channels, kernel_size=2)
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")

        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.fc = nn.Linear(hidden_channels * 10, 10)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
