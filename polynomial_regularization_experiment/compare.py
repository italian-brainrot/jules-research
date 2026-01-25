import optuna
import torch
from main import get_model, get_data, train_polynomial, train_adam

def objective_polynomial(trial):
    """Objective function for Optuna to optimize for the Polynomial Regularization method."""
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    reg_strength = trial.suggest_float('reg_strength', 1e-5, 1e-1, log=True)

    model = get_model()
    train_loader, test_loader = get_data()

    accuracy = train_polynomial(model, train_loader, test_loader, lr, reg_strength)
    return accuracy

def objective_adam(trial):
    """Objective function for Optuna to optimize for the baseline Adam optimizer."""
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = get_model()
    train_loader, test_loader = get_data()

    accuracy = train_adam(model, train_loader, test_loader, lr)
    return accuracy

if __name__ == '__main__':
    # --- Polynomial Regularization Study ---
    study_polynomial = optuna.create_study(direction='maximize')
    study_polynomial.optimize(objective_polynomial, n_trials=10)

    print("--- Polynomial Regularization Results ---")
    print(f"Best trial for Polynomial Regularization: {study_polynomial.best_trial.value}")
    print(f"Best params: {study_polynomial.best_params}")

    # --- Adam Baseline Study ---
    study_adam = optuna.create_study(direction='maximize')
    study_adam.optimize(objective_adam, n_trials=10)

    print("\n--- Adam Baseline Results ---")
    print(f"Best trial for Adam: {study_adam.best_trial.value}")
    print(f"Best params: {study_adam.best_params}")
