import torch
import torch.nn as nn
import torch.optim as optim
from .utils import get_model_and_data, train_and_evaluate
import numpy as np
import random
import os

# --- Configuration ---
POPULATION_SIZE = 20
N_GENERATIONS = 10
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
POLYNOMIAL_DEGREE = 3
N_EPOCHS = 5
LEARNING_RATE_SCALE = 1e-4

# --- Genetic Programming Core ---

class PolynomialSchedule:
    """Represents a polynomial learning rate schedule."""
    def __init__(self, coeffs):
        self.coeffs = np.array(coeffs, dtype=np.float32)

    def get_lr(self, epoch_normalized):
        """Calculate LR for a given normalized epoch (0 to 1)."""
        res = 0
        for c in reversed(self.coeffs):
            res = res * epoch_normalized + c
        return np.abs(res) * LEARNING_RATE_SCALE

    def __str__(self):
        return f"PolynomialSchedule(coeffs={self.coeffs})"

def create_individual():
    """Creates a random individual (polynomial)."""
    coeffs = np.random.randn(POLYNOMIAL_DEGREE + 1)
    return PolynomialSchedule(coeffs)

def crossover(parent1, parent2):
    """Performs one-point crossover."""
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1.coeffs) - 1)
        child1_coeffs = np.concatenate([parent1.coeffs[:point], parent2.coeffs[point:]])
        child2_coeffs = np.concatenate([parent2.coeffs[:point], parent1.coeffs[point:]])
        return PolynomialSchedule(child1_coeffs), PolynomialSchedule(child2_coeffs)
    return parent1, parent2

def mutate(individual):
    """Mutates an individual by adding Gaussian noise."""
    mutated_coeffs = individual.coeffs.copy()
    for i in range(len(mutated_coeffs)):
        if random.random() < MUTATION_RATE:
            mutated_coeffs[i] += np.random.normal(0, 0.1)
    return PolynomialSchedule(mutated_coeffs)

# --- Fitness Evaluation ---

def evaluate_fitness(individual, model_template, train_loader, val_loader):
    """Trains a model with the given LR schedule and returns its validation accuracy."""
    val_accuracies = train_and_evaluate(model_template, train_loader, val_loader, individual.get_lr, N_EPOCHS)
    return val_accuracies[-1]

# --- Main Evolution Loop ---

def tournament_selection(population, fitnesses):
    """Selects an individual using tournament selection."""
    selected_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best_index = -1
    best_fitness = -1
    for index in selected_indices:
        if fitnesses[index] > best_fitness:
            best_fitness = fitnesses[index]
            best_index = index
    return population[best_index]

def main():
    print("Initializing population...")
    population = [create_individual() for _ in range(POPULATION_SIZE)]

    # Get a consistent model and data for all evaluations
    model_template, train_loader, val_loader = get_model_and_data()

    for generation in range(N_GENERATIONS):
        print(f"\n--- Generation {generation+1}/{N_GENERATIONS} ---")

        print("Evaluating fitness...")
        fitnesses = [evaluate_fitness(ind, model_template, train_loader, val_loader) for ind in population]

        best_fitness_idx = np.argmax(fitnesses)
        best_individual = population[best_fitness_idx]
        print(f"Best Fitness: {fitnesses[best_fitness_idx]:.4f}")
        print(f"Best Individual: {best_individual}")

        new_population = [best_individual] # Elitism

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2)

            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population

    print("\nEvolution finished!")
    final_fitnesses = [evaluate_fitness(ind, model_template, train_loader, val_loader) for ind in population]
    best_final_idx = np.argmax(final_fitnesses)
    best_overall_individual = population[best_final_idx]

    print(f"\nBest overall individual found:")
    print(best_overall_individual)
    print(f"With final fitness: {final_fitnesses[best_final_idx]:.4f}")

    # Save the best individual's coefficients
    output_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(output_dir, 'best_coeffs.npy'), best_overall_individual.coeffs)
    print(f"Saved best coefficients to {os.path.join(output_dir, 'best_coeffs.npy')}")

if __name__ == "__main__":
    # Set a seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()
