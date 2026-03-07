import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 2)
        self.activations = None

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x.retain_grad()
        self.activations = x
        x = self.fc2(x)
        return x

model = SimpleModel()
x = torch.randn(3, 5)
y = torch.tensor([0, 1, 0])

logits = model(x)
loss = F.cross_entropy(logits, y, reduction='mean')

# Method 1: autograd.grad
grads_autograd = torch.autograd.grad(loss, model.activations, create_graph=True)[0]

# Method 2: manual per-sample grad
grads_manual = []
for i in range(3):
    model.zero_grad()
    l = F.cross_entropy(model(x[i:i+1]), y[i:i+1], reduction='mean')
    g = torch.autograd.grad(l, model.activations, retain_graph=True)[0]
    grads_manual.append(g)
grads_manual = torch.cat(grads_manual, dim=0)

print("Autograd grads (scaled by 1/B because of mean loss):")
print(grads_autograd)
print("Manual per-sample grads:")
print(grads_manual)

# Check if grads_autograd * 3 == grads_manual
# Wait, Method 2 uses mean loss on a batch of size 1, so it's not scaled by 1/3.
# So grads_autograd should be grads_manual / 3.
diff = torch.abs(grads_autograd - grads_manual / 3).max()
print(f"Difference: {diff.item()}")
