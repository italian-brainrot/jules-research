import torch
import numpy as np

def get_hoc_features(x):
    """
    x: (B, L) tensor
    returns: (B, F) tensor of Higher-Order Correlation features
    """
    B, L = x.shape
    # Ensure float64 for feature extraction to maintain precision
    orig_dtype = x.dtype
    x = x.to(torch.float64)

    X = torch.fft.rfft(x, n=L) # (B, L//2 + 1)
    X_len = X.shape[1]

    def get_X(k):
        # k is from 0 to L-1
        k = k % L
        if k < X_len:
            return X[:, k]
        else:
            return torch.conj(X[:, L - k])

    # 1. First order: Mean
    mean = X[:, 0].real.unsqueeze(1) / L

    # 2. Second order: Power Spectrum
    power_spectrum = torch.abs(X)**2 # (B, L//2 + 1)

    # 3. Third order: Bispectrum
    # We use a broader range for the bispectrum
    bispectrum_list = []

    # Principal domain for discrete bispectrum of real signal:
    # 0 <= k2 <= k1 <= L/2 and k1 + k2 <= L (already covered by k1 <= L/2)
    # Actually it is 0 <= k2 <= k1 <= L/2 AND 2*k1 + k2 <= L
    # Wait, let's just use 0 <= k2 <= k1 <= L/2.
    # This covers the whole triangle (0,0), (L/2, 0), (L/2, L/2)? No, (L/2, L/2) is not possible as k1+k2 would be L.

    for k1 in range(X_len):
        for k2 in range(k1 + 1):
            # B(k1, k2) = X(k1) * X(k2) * conj(X(k1+k2))
            k3 = (k1 + k2) % L
            b = X[:, k1] * X[:, k2] * torch.conj(get_X(k3))

            bispectrum_list.append(b.real.unsqueeze(1))
            bispectrum_list.append(b.imag.unsqueeze(1))

    features = torch.cat([mean, power_spectrum, *bispectrum_list], dim=1)
    return features.to(orig_dtype)

if __name__ == "__main__":
    # Test translation invariance
    L = 40
    x = torch.randn(2, L, dtype=torch.float64)
    x_shift = torch.roll(x, shifts=5, dims=1)

    f1 = get_hoc_features(x)
    f2 = get_hoc_features(x_shift)

    diff = torch.norm(f1 - f2)
    print(f"Feature dimension: {f1.shape[1]}")
    print(f"Difference after shift: {diff.item()}")
    assert diff < 1e-10
