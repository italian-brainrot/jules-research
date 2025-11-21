import numpy as np
from gp_framework import (
    initialize_population,
    evaluate_fitness,
    selection,
    crossover,
    mutation,
)

# Load the dataset
data = np.load('logistic_regression_dataset.npz')
X = data['X']
y = data['y']

# Set GP parameters
POPULATION_SIZE = 50
MAX_DEPTH = 4
GENERATIONS = 10
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.2

# Initialize the population
population = initialize_population(POPULATION_SIZE, MAX_DEPTH)

# Main GP loop
for generation in range(GENERATIONS):
    print(f"Generation {generation + 1}/{GENERATIONS}")

    # Evaluate fitness
    fitnesses = evaluate_fitness(population, X, y, X.shape[1])

    # Select the best individuals
    selected_population = selection(population, fitnesses)

    # Create the next generation
    next_population = []
    for i in range(0, POPULATION_SIZE, 2):
        parent1 = selected_population[i]
        parent2 = selected_population[i+1]

        if np.random.rand() < CROSSOVER_RATE:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        if np.random.rand() < MUTATION_RATE:
            child1 = mutation(child1, MAX_DEPTH)
        if np.random.rand() < MUTATION_RATE:
            child2 = mutation(child2, MAX_DEPTH)

        next_population.extend([child1, child2])

    population = next_population

# Find the best individual
fitnesses = evaluate_fitness(population, X, y, X.shape[1])
best_individual_index = np.argmin(fitnesses)
best_individual = population[best_individual_index]

def print_tree(node, indent=""):
    print(f"{indent}{node.value}")
    for child in node.children:
        print_tree(child, indent + "  ")

import pickle

print("\nBest optimizer found:")
print_tree(best_individual)

# Save the best individual
with open('best_optimizer.pkl', 'wb') as f:
    pickle.dump(best_individual, f)
