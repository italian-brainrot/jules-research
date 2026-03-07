import torch
import torch.nn.functional as F

def get_logit_gradients(model, x, y):
    """
    Computes per-sample gradients of the cross-entropy loss with respect to logits.
    Actually, we want the gradient of the loss with respect to inputs or weights?
    Wait, the hypothesis said: "cosine similarity between the logit gradients (gradients of the loss with respect to the pre-softmax activations) of the two samples".

    If L is the cross-entropy loss, and z is the logit vector.
    dL/dz_i = p_i - y_i where p is the softmax output and y is the one-hot target.
    This is very cheap to compute! No need for torch.func if we just need dL/dz.
    """
    model.eval() # We just need the logits
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        y_onehot = F.one_hot(y, num_classes=logits.size(1)).float()
        logit_grads = probs - y_onehot
    model.train()
    return logit_grads

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def lgam_mixup_data(model, x, y, alpha=1.0, gamma=1.0):
    """
    Logit-Gradient-Agreement Mixup (LGAM)
    """
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    logit_grads = get_logit_gradients(model, x, y)

    # Compute cosine similarity between g_i and g_j where j is index[i]
    g_i = logit_grads
    g_j = logit_grads[index]

    # Normalize for cosine similarity
    g_i_norm = F.normalize(g_i, p=2, dim=1)
    g_j_norm = F.normalize(g_j, p=2, dim=1)

    cos_sim = (g_i_norm * g_j_norm).sum(dim=1) # (batch_size,)

    # Map cos_sim from [-1, 1] to [0, 1] (optional, but hypothesis suggested s_ij = cos(g_i, g_j))
    # If s_ij is large, we want more mixing (large lambda).
    # Wait, lambda is usually around 0.5 in Mixup? No, it's from Beta(alpha, alpha).

    # If alpha=1.0, Beta is uniform.
    # We can adjust alpha per pair, but Beta sample is usually scalar for whole batch.
    # To do it per sample, we need per-sample lambda.

    lams = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(x.device)

    # Adjustment factor: ( (1 + cos_sim)/2 )^gamma
    # High agreement -> higher mixing potential? Or should it be:
    # High agreement -> these samples are already similar in how they affect the model,
    # so mixing them is "safe" and we can push them further towards the manifold.
    # Low agreement -> they are conflicting, mixing might be destructive.

    # s_ij = cos_sim
    # We want to favor lambda near 0.5 when s_ij is high?
    # Standard mixup lambda is sampled once for the batch.
    # For LGAM, let's use per-sample lambda adjusted by agreement.

    # If s_ij is high, we keep lam as is (or encourage it to be closer to 0.5).
    # If s_ij is low, we push lam towards 0 or 1 (less mixing).

    # Actually, a simpler way:
    # lam_i' = 0.5 + (lam_i - 0.5) * (1 - agreement)^gamma
    # If agreement is 1, lam_i' = 0.5 (max mixing).
    # If agreement is -1 (or 0), lam_i' = lam_i.

    # Agreement s_ij in [0, 1]
    agreement = (cos_sim + 1) / 2

    # If agreement is high, we allow more mixing (push lams towards 0.5)
    # If agreement is low, we keep mixing as is (likely less mixing if alpha is small)
    adj = agreement.pow(gamma)
    lams = 0.5 + (lams - 0.5) * (1 - adj)

    lams = lams.view(-1, 1)
    mixed_x = lams * x + (1 - lams) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lams

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    if isinstance(lam, torch.Tensor):
        # Per-sample lambda. We need per-sample loss here.
        # Use functional cross_entropy to avoid modifying criterion state
        loss_a = F.cross_entropy(pred, y_a, reduction='none')
        loss_b = F.cross_entropy(pred, y_b, reduction='none')

        # lam is (batch_size, 1), loss_a/b are (batch_size,)
        loss = (lam.squeeze() * loss_a + (1 - lam.squeeze()) * loss_b).mean()
    else:
        loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return loss
