import torch
import numpy as np
from sklearn.linear_model import Ridge

class RationalFeatureExpansion:
    """
    Fits a rational function f(x) = P(x) / Q(x) where P and Q are linear in random features.
    The optimization is done via Iterative Rational Least Squares (Sanathanan-Koerner iteration).
    """
    def __init__(self, num_features, sigma=1.0, reg=1e-3, num_iter=3, device='cpu'):
        self.num_features = num_features
        self.sigma = sigma
        self.reg = reg
        self.num_iter = num_iter
        self.device = device
        self.W = None
        self.b = None
        self.wa = None # Weights for numerator (C, M+1)
        self.wb = None # Weights for denominator (C, M+1)

    def _get_rff(self, x):
        if self.W is None:
            D = x.shape[1]
            torch.manual_seed(42)
            self.W = torch.randn(self.num_features // 2, D, device=self.device) * self.sigma
            self.b = torch.rand(self.num_features // 2, 1, device=self.device) * 2 * np.pi

        proj = x @ self.W.T + self.b.T
        rff = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        # Add bias
        ones = torch.ones(x.shape[0], 1, device=self.device)
        return torch.cat([rff, ones], dim=-1)

    def fit(self, X, y):
        # X: (N, D), y: (N, C) - one-hot or regression targets
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        X = X.to(self.device).float()
        y = y.to(self.device).float()
        N, C = y.shape

        phi = self._get_rff(X) # (N, M+1)
        M_plus_1 = phi.shape[1]

        self.wa = []
        self.wb = []

        for c in range(C):
            yc = y[:, c]
            Q_prev = torch.ones(N, device=self.device)
            w_P, w_Q = None, None

            for i in range(self.num_iter):
                # Solving for w_P, w_Q in: (w_P.T * phi) / Q_prev - (yc * w_Q.T * phi) / Q_prev = yc / Q_prev
                inv_Q = 1.0 / torch.clamp(torch.abs(Q_prev), min=1e-5)

                phi_scaled = phi * inv_Q.unsqueeze(1)
                yc_phi_scaled = (yc.unsqueeze(1) * phi) * inv_Q.unsqueeze(1)

                A = torch.cat([phi_scaled, -yc_phi_scaled], dim=1) # (N, 2*(M+1))
                target = yc * inv_Q

                clf = Ridge(alpha=self.reg, fit_intercept=False)
                clf.fit(A.cpu().numpy(), target.cpu().numpy())
                weights = torch.tensor(clf.coef_, dtype=torch.float32, device=self.device)

                w_P = weights[:M_plus_1]
                w_Q = weights[M_plus_1:]
                Q_prev = phi @ w_Q + 1.0

            self.wa.append(w_P)
            self.wb.append(w_Q)

        self.wa = torch.stack(self.wa)
        self.wb = torch.stack(self.wb)

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device).float()
        phi = self._get_rff(X)

        P = phi @ self.wa.T # (N, C)
        Q = 1.0 + phi @ self.wb.T # (N, C)

        probs = P / torch.clamp(Q, min=1e-5)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return torch.argmax(probs, dim=1)

class ELM:
    """Standard Extreme Learning Machine (just the numerator P(x) with Q(x)=1)."""
    def __init__(self, num_features, sigma=1.0, reg=1e-3, device='cpu'):
        self.num_features = num_features
        self.sigma = sigma
        self.reg = reg
        self.device = device
        self.W = None
        self.b = None
        self.wa = None

    def _get_rff(self, x):
        if self.W is None:
            D = x.shape[1]
            torch.manual_seed(42)
            self.W = torch.randn(self.num_features // 2, D, device=self.device) * self.sigma
            self.b = torch.rand(self.num_features // 2, 1, device=self.device) * 2 * np.pi

        proj = x @ self.W.T + self.b.T
        rff = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        ones = torch.ones(x.shape[0], 1, device=self.device)
        return torch.cat([rff, ones], dim=-1)

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        X = X.to(self.device).float()
        y = y.to(self.device).float()

        phi = self._get_rff(X)
        clf = Ridge(alpha=self.reg, fit_intercept=False)
        clf.fit(phi.cpu().numpy(), y.cpu().numpy())
        self.wa = torch.tensor(clf.coef_, dtype=torch.float32, device=self.device)

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device).float()
        phi = self._get_rff(X)
        return phi @ self.wa.T

    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1)
