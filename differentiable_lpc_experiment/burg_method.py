import torch
import torch.nn as nn

def burg_method(x, order):
    """
    Differentiable implementation of Burg's method for estimating LPC coefficients.
    x: (batch, length)
    order: scalar int
    Returns: (batch, order) coefficients (a_1, a_2, ..., a_p)
    where x_hat[t] = sum_{i=1}^p a_i * x[t-i]
    """
    batch_size, length = x.shape

    # f and b are forward and backward prediction errors
    f = x[:, 1:].clone() # (batch, length-1)
    b = x[:, :-1].clone() # (batch, length-1)

    # Initial denominator for k calculation
    den = torch.sum(f**2 + b**2, dim=1) # (batch,)

    a_list = [x.new_zeros((batch_size, order))]

    for m in range(1, order + 1):
        # Calculate reflection coefficient k_m
        num = -2.0 * torch.sum(f * b, dim=1)

        # Avoid division by zero
        k = num / (den + 1e-10)
        k = k.unsqueeze(1) # (batch, 1)

        # Clip k to ensure stability |k| < 1
        k = torch.clamp(k, -0.999, 0.999)

        # Update LPC coefficients a
        # a_m[m-1] = k_m
        # a_m[i] = a_{m-1}[i] + k_m * a_{m-1}[m-1-i] for i < m-1

        a_prev = a_list[-1]
        a_curr = a_prev.clone()
        a_curr[:, m-1] = k.squeeze(1)
        if m > 1:
            # Reversing a_prev up to m-1
            # a_curr[:, :m-1] = a_prev[:, :m-1] + k * a_prev[:, :m-1].flip(dims=[1])
            # wait, it is a_prev[:, :m-1] + k * a_prev[:, 0:m-1].flip(dims=[1])
            a_curr[:, :m-1] = a_prev[:, :m-1] + k * torch.flip(a_prev[:, :m-1], dims=[1])

        a_list.append(a_curr)

        if m < order:
            # Update f and b for next step
            f_old = f
            b_old = b

            f = f_old[:, 1:] + k * b_old[:, 1:]
            b = b_old[:, :-1] + k * f_old[:, :-1]

            # Update denominator efficiently
            den = den * (1.0 - k.squeeze(1)**2) - f_old[:, 0]**2 - b_old[:, -1]**2
            den = torch.clamp(den, min=1e-10)

    return a_list[-1]

def levinson_recursion(r, order):
    """
    Differentiable Levinson-Durbin recursion.
    r: (batch, order + 1) - Autocorrelation coefficients
    order: int
    Returns: (batch, order) - LPC coefficients
    """
    batch_size = r.shape[0]
    a_list = [r.new_zeros((batch_size, order))]
    e = r[:, 0]

    for i in range(order):
        # i is the current order of the recursion (0 to order-1)

        a_prev = a_list[-1]
        if i == 0:
            num = r[:, 1]
        else:
            # sum_{j=0}^{i-1} a_{i-1,j} r_{i-j}
            # a_prev is order (batch, order), valid coefficients are in :i
            sum_val = torch.sum(a_prev[:, :i] * torch.flip(r[:, 1:i+1], dims=[1]), dim=1)
            num = r[:, i+1] - sum_val

        k = num / (e + 1e-10)
        k = torch.clamp(k, -0.999, 0.999)

        a_curr = a_prev.clone()
        a_curr[:, i] = k
        if i > 0:
            a_curr[:, :i] = a_prev[:, :i] - k.unsqueeze(1) * torch.flip(a_prev[:, :i], dims=[1])

        a_list.append(a_curr)
        e = e * (1.0 - k**2)

    return a_list[-1]

def get_autocorrelation(x, order):
    batch_size, length = x.shape
    r = []
    for i in range(order + 1):
        if i == 0:
            r.append(torch.sum(x * x, dim=1))
        else:
            r.append(torch.sum(x[:, i:] * x[:, :-i], dim=1))
    return torch.stack(r, dim=1)

def lpc_from_autocorr(x, order):
    r = get_autocorrelation(x, order)
    return levinson_recursion(r, order)

class LPCLayer(nn.Module):
    def __init__(self, order, method='burg'):
        super().__init__()
        self.order = order
        self.method = method

    def forward(self, x):
        if self.method == 'burg':
            return burg_method(x, self.order)
        elif self.method == 'levinson':
            return lpc_from_autocorr(x, self.order)
        else:
            raise ValueError(f"Unknown method {self.method}")
