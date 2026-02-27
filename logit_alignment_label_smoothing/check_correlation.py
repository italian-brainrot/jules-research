import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call

def test_alignment_correlation():
    B, Din, Dout = 64, 100, 10
    x = torch.randn(B, Din)
    y = torch.randint(0, Dout, (B,))

    model = nn.Sequential(
        nn.Linear(Din, 50),
        nn.ReLU(),
        nn.Linear(50, Dout)
    )

    params = dict(model.named_parameters())

    def compute_loss(params, x, y):
        logits = functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)
        loss = F.cross_entropy(logits, y)
        return loss

    # Per-sample gradients w.r.t. parameters
    per_sample_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(params, x, y)

    # Flatten and concatenate grads for each sample
    def flatten_grads(grads):
        flat = []
        for p in grads.values():
            flat.append(p.flatten(1))
        return torch.cat(flat, dim=1)

    flat_grads = flatten_grads(per_sample_grads)
    mean_grad = flat_grads.mean(dim=0, keepdim=True)

    # Weight-gradient alignment
    cos = nn.CosineSimilarity(dim=1)
    s_weight = cos(flat_grads, mean_grad)

    # Logit-gradient alignment
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    y_onehot = F.one_hot(y, num_classes=Dout).float()
    g_logits = probs - y_onehot
    mean_g_logits = g_logits.mean(dim=0, keepdim=True)
    s_logit = cos(g_logits, mean_g_logits)

    correlation = torch.corrcoef(torch.stack([s_weight, s_logit]))[0, 1]
    print(f"Correlation between weight-gradient alignment and logit-gradient alignment: {correlation.item():.4f}")

if __name__ == "__main__":
    test_alignment_correlation()
