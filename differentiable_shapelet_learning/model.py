import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableShapeletLayer(nn.Module):
    def __init__(self, in_channels, num_shapelets, shapelet_length, temperature=1.0):
        super(DifferentiableShapeletLayer, self).__init__()
        self.in_channels = in_channels
        self.num_shapelets = num_shapelets
        self.shapelet_length = shapelet_length
        self.temperature = temperature

        # Shapelets are (num_shapelets, in_channels, shapelet_length)
        self.shapelets = nn.Parameter(torch.randn(num_shapelets, in_channels, shapelet_length))

    def forward(self, x):
        # x: (batch_size, in_channels, signal_length)
        batch_size, in_channels, signal_length = x.shape

        # Unfold x into sliding windows
        # x_unfolded: (batch_size, in_channels, num_windows, shapelet_length)
        x_unfolded = x.unfold(2, self.shapelet_length, 1)
        num_windows = x_unfolded.shape[2]

        # Reorder to (batch_size, 1, num_windows, in_channels, shapelet_length)
        # Permute: (batch, in_channels, num_windows, shapelet_length) -> (batch, num_windows, in_channels, shapelet_length)
        x_unfolded = x_unfolded.permute(0, 2, 1, 3).unsqueeze(1)

        # shapelets: (1, num_shapelets, 1, in_channels, shapelet_length)
        shapelets = self.shapelets.unsqueeze(0).unsqueeze(2)

        # Compute squared Euclidean distance
        # Subtraction broadcasts to (batch_size, num_shapelets, num_windows, in_channels, shapelet_length)
        # distances: (batch_size, num_shapelets, num_windows)
        distances = torch.sum((x_unfolded - shapelets)**2, dim=(3, 4))

        # Soft-min pooling over windows to find the best match for each shapelet
        # We use -distances / temperature for soft-min
        soft_min_weights = F.softmax(-distances / self.temperature, dim=2)
        pooled_distances = torch.sum(distances * soft_min_weights, dim=2)

        return pooled_distances

class ShapeletNetwork(nn.Module):
    def __init__(self, in_channels, num_shapelets, shapelet_length, num_classes, hidden_dim=64, temperature=1.0):
        super(ShapeletNetwork, self).__init__()
        self.shapelet_layer = DifferentiableShapeletLayer(in_channels, num_shapelets, shapelet_length, temperature)
        self.classifier = nn.Sequential(
            nn.Linear(num_shapelets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x should be (batch_size, 40) for mnist1d, we need to add channel dim
        if x.dim() == 2:
            x = x.unsqueeze(1)

        features = self.shapelet_layer(x)
        logits = self.classifier(features)
        return logits

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class Conv1dBaseline(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, num_classes, hidden_dim=64):
        super(Conv1dBaseline, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.conv(x)
        logits = self.classifier(features)
        return logits
