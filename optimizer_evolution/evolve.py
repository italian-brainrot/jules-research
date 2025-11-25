import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import mnist1d.data as mnist1d_data
from gp import Individual, random_expression, subtree_crossover, subtree_mutation, primitives, terminals
from optimizer import EvolvedOptimizer

# Define the neural network for mnist1d
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(40, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# Function to evaluate an individual
def evaluate(individual, train_loader, epochs=1, lr=0.01):
    model = SimpleNN()
    loss_fn = nn.CrossEntropyLoss()

    # Create the optimizer from the individual's expression
    try:
        update_rule_str = individual.to_pytorch()
        optimizer = EvolvedOptimizer(model.parameters(), update_rule_str, lr=lr)
    except Exception as e:
        print(f"Error creating optimizer: {e}")
        return float('inf')

    # Training loop
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx >= 20: # Limit the number of batches for faster evaluation
            break

    return total_loss / 20

# Main evolution loop
def main():
    # Load mnist1d data
    args = mnist1d_data.get_dataset_args()
    data = mnist1d_data.get_dataset(args)
    X_train, y_train = torch.from_numpy(data['x']).float(), torch.from_numpy(data['y']).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # GP parameters
    population_size = 20
    num_generations = 10
    crossover_rate = 0.7
    mutation_rate = 0.2
    max_depth = 5

    # Initialize population
    population = [Individual(random_expression(primitives, terminals, max_depth)) for _ in range(population_size)]

    # Evolution loop
    for gen in range(num_generations):
        print(f"Generation {gen+1}/{num_generations}")

        # Evaluate fitness
        for ind in population:
            ind.fitness = evaluate(ind, train_loader)
            print(f"  Individual: {str(ind)}, Fitness: {ind.fitness}")

        # Sort population by fitness
        population.sort(key=lambda x: x.fitness)
        print(f"Best individual: {population[0]}, Fitness: {population[0].fitness}")

        # Create the next generation
        new_population = []

        # Elitism
        new_population.extend(population[:int(0.1 * population_size)])

        # Crossover and Mutation
        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                p1, p2 = random.sample(population[:int(0.5*population_size)], 2) # Tournament selection
                c1, c2 = subtree_crossover(p1, p2)
                new_population.extend([c1, c2])
            if random.random() < mutation_rate:
                p = random.choice(population[:int(0.5*population_size)])
                new_population.append(subtree_mutation(p, primitives, terminals, max_depth))

        population = new_population[:population_size]

    # Print the best optimizer
    best_optimizer = population[0]
    print(f"\nBest optimizer found: {best_optimizer}")
    print(f"Pytorch code: {best_optimizer.to_pytorch()}")

if __name__ == "__main__":
    main()
