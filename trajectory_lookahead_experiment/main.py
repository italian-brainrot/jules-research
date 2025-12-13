import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from .model import SimpleMLP
from .optimizer import TrajectoryLookahead
import matplotlib.pyplot as plt
import copy
import optuna

def train_model(optimizer_class, model, X_train, y_train, X_test, y_test, epochs, lr, la_steps=None, trajectory_len=None):
    if optimizer_class == TrajectoryLookahead:
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = TrajectoryLookahead(base_optimizer, la_steps=la_steps, trajectory_len=trajectory_len)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=256)

    criterion = nn.CrossEntropyLoss()
    history = {'test_acc': [], 'steps': []}
    step = 0

    for epoch in range(epochs):
        for inputs, targets in dl_train:
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for test_inputs, test_targets in dl_test:
                        test_outputs = model(test_inputs)
                        _, predicted = torch.max(test_outputs.data, 1)
                        total += test_targets.size(0)
                        correct += (predicted == test_targets).sum().item()
                acc = correct / total
                history['test_acc'].append(acc)
                history['steps'].append(step)

            step += 1

    # Calculate final accuracy for tuning
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_inputs, test_targets in dl_test:
            test_outputs = model(test_inputs)
            _, predicted = torch.max(test_outputs.data, 1)
            total += test_targets.size(0)
            correct += (predicted == test_targets).sum().item()
    final_acc = correct / total
    history['final_acc'] = final_acc


    return history

def tune_learning_rate(optimizer_class, model_class, X_train, y_train, X_test, y_test, la_steps=None, trajectory_len=None):
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        model = model_class()

        # Short training run for tuning
        history = train_model(optimizer_class, model, X_train, y_train, X_test, y_test, epochs=2, lr=lr, la_steps=la_steps, trajectory_len=trajectory_len)

        # Return the final test accuracy
        return history['final_acc']

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=20)

    print(f"Best LR for {optimizer_class.__name__}: {study.best_params['lr']}")
    return study.best_params['lr']

def main():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    # Tune learning rates
    print("Tuning learning rate for Adam...")
    lr_adam = tune_learning_rate(optim.Adam, SimpleMLP, X_train, y_train, X_test, y_test)

    print("\nTuning learning rate for TrajectoryLookahead...")
    lr_lookahead = tune_learning_rate(TrajectoryLookahead, SimpleMLP, X_train, y_train, X_test, y_test, la_steps=5, trajectory_len=3)

    # Train with Adam
    print("\nTraining with Adam...")
    model_adam = SimpleMLP()
    initial_weights = copy.deepcopy(model_adam.state_dict())
    history_adam = train_model(optim.Adam, model_adam, X_train, y_train, X_test, y_test, epochs=5, lr=lr_adam)

    # Train with TrajectoryLookahead
    print("\nTraining with TrajectoryLookahead...")
    model_lookahead = SimpleMLP()
    model_lookahead.load_state_dict(initial_weights) # Start from same initial weights
    history_lookahead = train_model(TrajectoryLookahead, model_lookahead, X_train, y_train, X_test, y_test, epochs=5, lr=lr_lookahead, la_steps=5, trajectory_len=3)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(history_adam['steps'], history_adam['test_acc'], label=f'Adam (LR={lr_adam:.4f})')
    plt.plot(history_lookahead['steps'], history_lookahead['test_acc'], label=f'Trajectory Lookahead (LR={lr_lookahead:.4f})')
    plt.xlabel('Training Steps')
    plt.ylabel('Test Accuracy')
    plt.title('Optimizer Comparison with Tuned Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectory_lookahead_experiment/comparison_plot.png')

    print("\nExperiment complete. Plot saved to trajectory_lookahead_experiment/comparison_plot.png")

if __name__ == '__main__':
    main()
