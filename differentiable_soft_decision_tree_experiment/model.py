import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim, output_dim, depth=4, beta=1.0):
        super(SoftDecisionTree, self).__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inner_nodes = 2**depth - 1
        self.num_leaves = 2**depth

        self.inner_nodes = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(self.num_inner_nodes)
        ])

        self.leaf_nodes = nn.Parameter(torch.randn(self.num_leaves, output_dim))
        self.beta = beta

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Pre-calculate all gating values: (batch_size, num_inner_nodes)
        gatings = torch.stack([torch.sigmoid(self.beta * node(x)).squeeze(1) for node in self.inner_nodes], dim=1)

        # node_probs stores the probability of reaching each node
        # node 0 is root.
        # left child of i is 2*i + 1, right is 2*i + 2
        node_probs = [None] * (self.num_inner_nodes + self.num_leaves)
        node_probs[0] = torch.ones(batch_size, device=device)

        for i in range(self.num_inner_nodes):
            left_child = 2 * i + 1
            right_child = 2 * i + 2

            # Probability of reaching the current node i
            p_i = node_probs[i]

            # Gating probability at node i
            g_i = gatings[:, i]

            node_probs[left_child] = p_i * g_i
            node_probs[right_child] = p_i * (1 - g_i)

        leaf_probs = torch.stack(node_probs[self.num_inner_nodes:], dim=1) # (batch_size, num_leaves)

        # Each leaf has a prediction (e.g. class distribution)
        # For simplicity, we can just use the leaf_nodes directly if they are meant to be features,
        # or apply softmax if they are class probabilities.
        # Here we assume they could be anything, but let's use them as additive features or logits.

        out = torch.matmul(leaf_probs, self.leaf_nodes) # (batch_size, output_dim)
        return out

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SDTAugmentedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tree_depth=4):
        super(SDTAugmentedMLP, self).__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sdt = SoftDecisionTree(input_dim, output_dim=hidden_dim, depth=tree_depth)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        mlp_feat = self.mlp_layers(x)
        sdt_feat = self.sdt(x)
        return self.fc(mlp_feat + sdt_feat)

class SDTClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, depth=5):
        super(SDTClassifier, self).__init__()
        self.sdt = SoftDecisionTree(input_dim, output_dim, depth=depth)

    def forward(self, x):
        return self.sdt(x)
