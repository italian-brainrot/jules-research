import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .model import SimpleCNN
from .optimizer import ConjugateGradient, EigenConjugateGradient

def main():
    # Training settings
    batch_size = 64
    epochs = 1
    lr = 0.01

    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    # Use a subset of the data to speed up training
    train_subset = torch.utils.data.Subset(train_dataset, range(100 * 64))
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Models
    cg_model = SimpleCNN()
    eigen_cg_model = SimpleCNN()
    eigen_cg_model.load_state_dict(cg_model.state_dict())

    # Optimizers
    cg_optimizer = ConjugateGradient(cg_model.parameters(), lr=lr)
    eigen_cg_optimizer = EigenConjugateGradient(eigen_cg_model.parameters(), lr=lr)

    # Disable GPU usage
    torch.cuda.is_available = lambda : False

    # Loss function
    loss_fn = F.nll_loss

    print("Training with standard Conjugate Gradient...")
    for epoch in range(1, epochs + 1):
        train(cg_model, train_loader, cg_optimizer, loss_fn, epoch, is_eigen_cg=False)
        test(cg_model, test_loader, loss_fn)

    print("\nTraining with Eigen-Conjugate Gradient...")
    for epoch in range(1, epochs + 1):
        train(eigen_cg_model, train_loader, eigen_cg_optimizer, loss_fn, epoch, is_eigen_cg=True)
        test(eigen_cg_model, test_loader, loss_fn)

def train(model, train_loader, optimizer, loss_fn, epoch, is_eigen_cg=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward(create_graph=True)
            return loss

        if is_eigen_cg:
            # Create a dataloader with a single batch for the eigen calculation
            eigen_dataloader = [(data, target)]
            optimizer.step(closure, model, eigen_dataloader, loss_fn)
        else:
            optimizer.step(closure)

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {closure().item():.6f}')

def test(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == '__main__':
    main()
