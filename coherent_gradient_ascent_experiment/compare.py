import torch
from mnist1d.data import make_dataset, get_dataset_args

def load_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    return X_train, y_train, X_test, y_test

import torch.nn as nn
from torch.optim.optimizer import Optimizer

class CGD(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, gamma=0.5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma value: {gamma}, must be in [0, 1)")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, gamma=gamma)
        super(CGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('CGA does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_grad = state['prev_grad']
                beta1, beta2 = group['betas']

                state['step'] += 1

                lr_scale = 1.0
                if state['step'] > 1 and torch.linalg.norm(prev_grad) > 0:
                    flat_grad = grad.flatten()
                    flat_prev_grad = prev_grad.flatten()

                    norm_grad = torch.linalg.norm(flat_grad)
                    if norm_grad > 0:
                        cosine_sim = torch.dot(flat_grad, flat_prev_grad) / (norm_grad * torch.linalg.norm(flat_prev_grad) + group['eps'])
                        lr_scale = 1.0 + group['gamma'] * cosine_sim

                state['prev_grad'] = grad.clone().detach()

                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt()).add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1
                step_size *= lr_scale

                p.addcdiv_(exp_avg, denom, value=-step_size * bias_correction2 ** 0.5)

        return loss

def create_model():
    return nn.Sequential(
        nn.Linear(40, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

from light_dataloader import TensorDataLoader
from torch.optim import Adam

def train(model, optimizer, dl_train):
    model.train()
    total_loss = 0
    for inputs, targets in dl_train:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dl_train)

def evaluate(model, dl_test):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dl_test:
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return total_loss / len(dl_test), accuracy

import optuna

def objective_adam(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    X_train, y_train, X_test, y_test = load_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    model = create_model()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(10):
        train(model, optimizer, dl_train)

    _, accuracy = evaluate(model, dl_test)
    return accuracy

def objective_cgd(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.0, 1.0)

    X_train, y_train, X_test, y_test = load_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    model = create_model()
    optimizer = CGD(model.parameters(), lr=lr, gamma=gamma)

    for epoch in range(10):
        train(model, optimizer, dl_train)

    _, accuracy = evaluate(model, dl_test)
    return accuracy

import matplotlib.pyplot as plt

def run_experiment(optimizer_class, optimizer_params, X_train, y_train, X_test, y_test):
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    model = create_model()
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    accuracies = []
    for epoch in range(20):
        train(model, optimizer, dl_train)
        _, accuracy = evaluate(model, dl_test)
        accuracies.append(accuracy)
    return accuracies

if __name__ == '__main__':
    study_adam = optuna.create_study(direction="maximize")
    study_adam.optimize(objective_adam, n_trials=10)
    best_params_adam = study_adam.best_trial.params

    study_cgd = optuna.create_study(direction="maximize")
    study_cgd.optimize(objective_cgd, n_trials=10)
    best_params_cgd = study_cgd.best_trial.params

    X_train, y_train, X_test, y_test = load_data()

    adam_accuracies = run_experiment(Adam, best_params_adam, X_train, y_train, X_test, y_test)
    cgd_accuracies = run_experiment(CGD, best_params_cgd, X_train, y_train, X_test, y_test)

    plt.figure(figsize=(10, 6))
    plt.plot(adam_accuracies, label="Adam")
    plt.plot(cgd_accuracies, label="CGD")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Adam vs. CGD")
    plt.legend()
    plt.grid(True)
    plt.savefig("coherent_gradient_ascent_experiment/comparison.png")
    print("\nPlot saved to coherent_gradient_ascent_experiment/comparison.png")
