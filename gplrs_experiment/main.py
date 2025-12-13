
import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import random
import copy
import pickle
from gp_core import create_random_expression, evaluate_expression, crossover, mutation

# --- HYPERPARAMETERS ---
POPULATION_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 5
ELITISM_SIZE = 2
NUM_EPOCHS_FITNESS = 5 # Number of epochs to train for fitness evaluation
LEARNING_RATE_CAP = 1.0 # Cap for the evolved learning rate

# --- DATASET ---
def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 4000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    return dl_train, dl_test

# --- NEURAL NETWORK ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.layers(x)

# --- FITNESS FUNCTION ---
def calculate_fitness(individual, train_loader, val_loader):
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Initial LR doesn't matter much
    criterion = nn.CrossEntropyLoss()

    total_loss = 0

    # Use a small initial LR for the very first step
    last_batch_loss = 0.01

    for epoch in range(NUM_EPOCHS_FITNESS):
        model.train()
        for inputs, targets in train_loader:
            # Evaluate the GP expression to get the current learning rate
            # Use the loss from the *previous* batch
            lr = evaluate_expression(individual, epoch=epoch, loss=last_batch_loss, val_loss=0)
            lr = max(0.0, min(lr, LEARNING_RATE_CAP)) # Clamp the LR

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Store the loss for the next iteration
            last_batch_loss = loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

    accuracy = correct / total
    return accuracy, # Fitness is a tuple

# --- GENETIC ALGORITHM ---
def tournament_selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
        winner = max(tournament, key=lambda x: x[1][0])
        selected.append(winner[0])
    return selected

def main():
    train_loader, val_loader = get_data()

    # 1. Initialize Population
    population = [create_random_expression(max_depth=4) for _ in range(POPULATION_SIZE)]

    print("Starting evolution...")

    for gen in range(GENERATIONS):
        # 2. Evaluate Fitness
        fitnesses = [calculate_fitness(ind, train_loader, val_loader) for ind in population]

        best_fitness_in_gen = max(fitnesses)[0]
        avg_fitness = sum(f[0] for f in fitnesses) / len(fitnesses)
        print(f"Generation {gen+1}/{GENERATIONS} - Best Fitness: {best_fitness_in_gen:.4f}, Avg Fitness: {avg_fitness:.4f}")

        # 3. Selection
        selected_population = tournament_selection(population, fitnesses)

        # 4. Crossover and Mutation
        next_population = []

        # Elitism
        sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1][0], reverse=True)
        for i in range(ELITISM_SIZE):
            next_population.append(sorted_population[i][0])

        # Create the rest of the new generation
        while len(next_population) < POPULATION_SIZE:
            p1, p2 = random.sample(selected_population, 2)

            if random.random() < CROSSOVER_RATE:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1, p2

            if random.random() < MUTATION_RATE:
                c1 = mutation(c1)
            if random.random() < MUTATION_RATE:
                c2 = mutation(c2)

            next_population.append(c1)
            if len(next_population) < POPULATION_SIZE:
                next_population.append(c2)

        population = next_population

    # Final Evaluation
    print("\nEvolution finished. Finding the best individual...")
    final_fitnesses = [calculate_fitness(ind, train_loader, val_loader) for ind in population]
    best_individual = max(zip(population, final_fitnesses), key=lambda x: x[1][0])

    print(f"\nBest individual found:")
    print(f"  Expression: {best_individual[0]}")
    print(f"  Fitness (Accuracy): {best_individual[1][0]:.4f}")

    # Save the best individual to a file
    with open("gplrs_experiment/best_expression.pkl", "wb") as f:
        pickle.dump(best_individual[0], f)
    print("Best expression saved to gplrs_experiment/best_expression.pkl")

if __name__ == "__main__":
    main()
