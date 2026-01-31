import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPBaseline(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, output_size=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class GRUBaseline(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=10):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

class NTMBase(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=10, mem_n=20, mem_w=16):
        super().__init__()
        self.mem_n = mem_n
        self.mem_w = mem_w
        self.hidden_size = hidden_size

        self.controller = nn.GRUCell(input_size + mem_w, hidden_size)
        self.read_head_params = nn.Linear(hidden_size, mem_w + 1 + 1 + 3 + 1)
        self.write_head_params = nn.Linear(hidden_size, mem_w + 1 + 1 + 3 + 1 + mem_w + mem_w)
        self.fc = nn.Linear(hidden_size, output_size)

        self.register_buffer("initial_state", torch.zeros(1, hidden_size))
        self.register_buffer("initial_mem", torch.zeros(1, mem_n, mem_w))
        self.register_buffer("initial_read", torch.zeros(1, mem_w))
        self.register_buffer("initial_weights", torch.zeros(1, mem_n))
        self.initial_weights.data.fill_(1.0 / mem_n)

        # Precompute roll indices
        self.register_buffer("idx_plus", torch.roll(torch.arange(mem_n), shifts=1, dims=0))
        self.register_buffer("idx_minus", torch.roll(torch.arange(mem_n), shifts=-1, dims=0))

    def get_address(self, params, prev_w, mem):
        k = params[:, :self.mem_w]
        beta = F.softplus(params[:, self.mem_w : self.mem_w + 1])
        g = torch.sigmoid(params[:, self.mem_w + 1 : self.mem_w + 2])
        s = F.softmax(params[:, self.mem_w + 2 : self.mem_w + 5], dim=-1)
        gamma = 1.0 + F.softplus(params[:, self.mem_w + 5 : self.mem_w + 6])

        cos_sim = F.cosine_similarity(k.unsqueeze(1), mem, dim=-1)
        w_c = F.softmax(beta * cos_sim, dim=-1)
        w_g = g * w_c + (1 - g) * prev_w

        w_s = s[:, 0:1] * w_g[:, self.idx_plus] + \
              s[:, 1:2] * w_g + \
              s[:, 2:3] * w_g[:, self.idx_minus]

        w_sharp = w_s ** gamma
        return w_sharp / (w_sharp.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        h = self.initial_state.repeat(batch_size, 1)
        mem = self.initial_mem.repeat(batch_size, 1, 1)
        read_vec = self.initial_read.repeat(batch_size, 1)
        w_r = self.initial_weights.repeat(batch_size, 1)
        w_w = self.initial_weights.repeat(batch_size, 1)

        for t in range(x.size(1)):
            inp = torch.cat([x[:, t, :], read_vec], dim=-1)
            h = self.controller(inp, h)
            w_r = self.get_address(self.read_head_params(h), w_r, mem)
            read_vec = torch.matmul(w_r.unsqueeze(1), mem).squeeze(1)

            w_params = self.write_head_params(h)
            w_w = self.get_address(w_params[:, :self.mem_w+6], w_w, mem)
            erase = torch.sigmoid(w_params[:, self.mem_w+6 : 2*self.mem_w+6])
            add = torch.tanh(w_params[:, 2*self.mem_w+6 : 3*self.mem_w+6])

            mem = mem * (1 - torch.matmul(w_w.unsqueeze(-1), erase.unsqueeze(1))) + \
                  torch.matmul(w_w.unsqueeze(-1), add.unsqueeze(1))

        return self.fc(h)

class DMS_NTM(NTMBase):
    def __init__(self, input_size=1, hidden_size=64, output_size=10, mem_n=20, mem_w=16):
        super().__init__(input_size, hidden_size, output_size, mem_n, mem_w)
        self.read_head_params = nn.Linear(hidden_size, mem_w + 5)
        self.write_head_params = nn.Linear(hidden_size, 3 * mem_w + 5)

        indices = torch.arange(mem_n).float()
        rel_indices = indices.view(mem_n, 1) - indices.view(1, mem_n)
        rel_indices = (rel_indices + mem_n/2) % mem_n - mem_n/2
        self.register_buffer("rel_indices", rel_indices)

    def get_address(self, params, prev_w, mem):
        k = params[:, :self.mem_w]
        beta = F.softplus(params[:, self.mem_w : self.mem_w + 1])
        g = torch.sigmoid(params[:, self.mem_w + 1 : self.mem_w + 2])
        delta_p = params[:, self.mem_w + 2 : self.mem_w + 3]
        sigma = F.softplus(params[:, self.mem_w + 3 : self.mem_w + 4]) + 0.1
        gamma = 1.0 + F.softplus(params[:, self.mem_w + 4 : self.mem_w + 5])

        cos_sim = F.cosine_similarity(k.unsqueeze(1), mem, dim=-1)
        w_c = F.softmax(beta * cos_sim, dim=-1)
        w_g = g * w_c + (1 - g) * prev_w

        kernel = torch.exp(- (self.rel_indices - delta_p.view(-1, 1, 1))**2 / (2 * sigma.view(-1, 1, 1)**2))
        kernel = kernel / (kernel.sum(dim=2, keepdim=True) + 1e-8)
        w_s = torch.bmm(kernel, w_g.unsqueeze(-1)).squeeze(-1)

        w_sharp = w_s ** gamma
        return w_sharp / (w_sharp.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        h = self.initial_state.repeat(batch_size, 1)
        mem = self.initial_mem.repeat(batch_size, 1, 1)
        read_vec = self.initial_read.repeat(batch_size, 1)
        w_r = self.initial_weights.repeat(batch_size, 1)
        w_w = self.initial_weights.repeat(batch_size, 1)

        for t in range(x.size(1)):
            inp = torch.cat([x[:, t, :], read_vec], dim=-1)
            h = self.controller(inp, h)
            w_r = self.get_address(self.read_head_params(h), w_r, mem)
            read_vec = torch.matmul(w_r.unsqueeze(1), mem).squeeze(1)

            w_params = self.write_head_params(h)
            w_w = self.get_address(w_params[:, :self.mem_w+5], w_w, mem)
            erase = torch.sigmoid(w_params[:, self.mem_w+5 : 2*self.mem_w+5])
            add = torch.tanh(w_params[:, 2*self.mem_w+5 : 3*self.mem_w+5])

            mem = mem * (1 - torch.matmul(w_w.unsqueeze(-1), erase.unsqueeze(1))) + \
                  torch.matmul(w_w.unsqueeze(-1), add.unsqueeze(1))

        return self.fc(h)
