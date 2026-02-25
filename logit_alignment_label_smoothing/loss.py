import torch
import torch.nn as nn
import torch.nn.functional as F

class LGALSLoss(nn.Module):
    def __init__(self, epsilon_max=0.1, gamma=1.0):
        super(LGALSLoss, self).__init__()
        self.epsilon_max = epsilon_max
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (B, C)
        # targets: (B,)
        B, C = logits.shape
        probs = F.softmax(logits, dim=1)
        y_oh = F.one_hot(targets, num_classes=C).float()

        # Logit-gradient: p - y
        g = probs - y_oh # (B, C)

        # Batch mean logit-gradient
        g_mean = g.mean(dim=0, keepdim=True) # (1, C)

        # Cosine similarity
        # Use small epsilon for stability
        g_norm = g.norm(dim=1, keepdim=True)
        g_mean_norm = g_mean.norm(dim=1, keepdim=True)

        s = (g * g_mean).sum(dim=1, keepdim=True) / (g_norm * g_mean_norm + 1e-8)
        s = s.clamp(-1.0, 1.0)

        # Adaptive label smoothing epsilon
        # Map s from [-1, 1] to [1, 0] for (1-s)/2
        epsilon = self.epsilon_max * torch.pow((1.0 - s) / 2.0, self.gamma)

        # Standard Cross Entropy
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1))

        # Uniform Cross Entropy (KL divergence from uniform is related to this)
        uniform_loss = -log_probs.mean(dim=1, keepdim=True)

        # Combined loss
        loss = (1.0 - epsilon) * ce_loss + epsilon * uniform_loss

        return loss.mean()

def fixed_label_smoothing_loss(logits, targets, epsilon=0.1):
    log_probs = F.log_softmax(logits, dim=1)
    ce_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1))
    uniform_loss = -log_probs.mean(dim=1, keepdim=True)
    loss = (1.0 - epsilon) * ce_loss + epsilon * uniform_loss
    return loss.mean()
