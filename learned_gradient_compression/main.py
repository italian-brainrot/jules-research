import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from .optimizer import LearnedGradientCompressionOptimizer
from .utils import get_mnist_loader, SimpleNN
import os
import argparse

def train(optimizer_name, model, train_loader, epochs=5, lr=1e-3, chunk_size=256):
    if optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'LGC':
        base_optimizer = Adam(model.parameters(), lr=lr)
        optimizer = LearnedGradientCompressionOptimizer(model.parameters(), base_optimizer, chunk_size=chunk_size)

    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    return losses

def main():
    parser = argparse.ArgumentParser(description='Optimizer Comparison')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--chunk_size', type=int, default=256, help='Chunk size for LGC optimizer')
    parser.add_argument('--subset_size', type=int, default=5000, help='Size of the training subset')
    args = parser.parse_args()

    # Create results directory
    if not os.path.exists('learned_gradient_compression/results'):
        os.makedirs('learned_gradient_compression/results')

    train_loader = get_mnist_loader(subset_size=args.subset_size)

    # Train with Adam
    model_adam = SimpleNN()
    losses_adam = train('Adam', model_adam, train_loader, epochs=args.epochs, lr=args.lr)

    # Train with Learned Gradient Compression
    model_lgc = SimpleNN()
    losses_lgc = train('LGC', model_lgc, train_loader, epochs=args.epochs, lr=args.lr, chunk_size=args.chunk_size)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(losses_adam, label='Adam')
    plt.plot(losses_lgc, label='Learned Gradient Compression')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.savefig('learned_gradient_compression/results/loss_comparison.png')
    plt.show()

if __name__ == '__main__':
    main()
