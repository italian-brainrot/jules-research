import torch
import torch.nn.functional as F
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from .model import MLP
from .optimizer import ConjugateGradient, EigenConjugateGradient

def main():
    # Training settings
    batch_size = 64
    epochs = 1

    # mnist1d Dataset
    defaults = get_dataset_args()
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=batch_size, shuffle=False)

    # Loss function
    loss_fn = F.nll_loss

    # Learning rate tuning
    print("Tuning learning rates...")
    best_cg_lr = tune_lr(MLP, train_loader, test_loader, loss_fn, ConjugateGradient)
    best_eigen_cg_lr = tune_lr(MLP, train_loader, test_loader, loss_fn, EigenConjugateGradient, is_eigen_cg=True, dataloader=train_loader)

    print(f"Best LR for CG: {best_cg_lr}")
    print(f"Best LR for Eigen-CG: {best_eigen_cg_lr}")

    # Models
    cg_model = MLP()
    eigen_cg_model = MLP()
    eigen_cg_model.load_state_dict(cg_model.state_dict())

    # Optimizers
    cg_optimizer = ConjugateGradient(cg_model.parameters(), lr=best_cg_lr)
    eigen_cg_optimizer = EigenConjugateGradient(eigen_cg_model.parameters(), lr=best_eigen_cg_lr)

    print("\nTraining with standard Conjugate Gradient...")
    for epoch in range(1, epochs + 1):
        train(cg_model, train_loader, cg_optimizer, loss_fn, epoch, is_eigen_cg=False)
        test(cg_model, test_loader, loss_fn)

    print("\nTraining with Eigen-Conjugate Gradient...")
    for epoch in range(1, epochs + 1):
        train(eigen_cg_model, train_loader, eigen_cg_optimizer, loss_fn, epoch, is_eigen_cg=True, dataloader=train_loader)
        test(eigen_cg_model, test_loader, loss_fn)

def tune_lr(model_class, train_loader, test_loader, loss_fn, optimizer_class, is_eigen_cg=False, dataloader=None):
    best_lr = 0
    best_accuracy = 0

    learning_rates = [0.001, 0.01, 0.1]

    for lr in learning_rates:
        model = model_class()
        optimizer = optimizer_class(model.parameters(), lr=lr)

        # Train for one epoch
        train(model, train_loader, optimizer, loss_fn, 1, is_eigen_cg, dataloader)

        # Evaluate on the test set
        accuracy = test(model, test_loader, loss_fn, silent=True)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr

    return best_lr

def train(model, train_loader, optimizer, loss_fn, epoch, is_eigen_cg=False, dataloader=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward(create_graph=True)
            return loss

        if is_eigen_cg:
            optimizer.step(closure, model, dataloader, loss_fn)
        else:
            optimizer.step(closure)

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.data[0])} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {closure().item():.6f}')

def test(model, test_loader, loss_fn, silent=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.data[0])
    accuracy = 100. * correct / len(test_loader.data[0])

    if not silent:
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.data[0])} '
              f'({accuracy:.0f}%)\n')

    return accuracy

if __name__ == '__main__':
    main()
