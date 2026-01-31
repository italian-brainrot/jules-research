import torch
import torch.nn as nn
import torch.nn.functional as F

class GreedyLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))

class GreedyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()

        prev_dim = input_dim
        for i in range(num_layers):
            self.layers.append(GreedyLayer(prev_dim, hidden_dim))
            self.heads.append(nn.Linear(hidden_dim, num_classes))
            prev_dim = hidden_dim

    def forward_layer(self, x, layer_idx):
        """Returns the hidden state of the layer_idx-th layer (0-indexed)."""
        h = x
        for i in range(layer_idx + 1):
            h = self.layers[i](h)
        return h

    def forward_head(self, x, layer_idx):
        """Returns the output of the head associated with layer_idx."""
        h = self.forward_layer(x, layer_idx)
        return self.heads[layer_idx](h)

    def forward_boost(self, x, layer_idx):
        """Returns the sum of outputs of heads up to layer_idx."""
        total_logits = 0
        h = x
        for i in range(layer_idx + 1):
            h = self.layers[i](h)
            total_logits = total_logits + self.heads[i](h)
        return total_logits

    def forward(self, x):
        """Standard forward pass: returns the output of the last head."""
        return self.forward_head(x, self.num_layers - 1)

    def forward_all_heads_sum(self, x):
        """Returns the sum of all heads."""
        return self.forward_boost(x, self.num_layers - 1)
