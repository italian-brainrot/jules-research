
import numpy as np
import random
import pickle

# Define the terminals and operators
TERMINALS = ['x', 'b']
OPERATORS = ['+', '-', '*', 'A*']
POPULATION_SIZE = 50
MAX_GENERATIONS = 20
MAX_DEPTH = 4
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
MAX_ITERATIONS = 25

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        if self.value in TERMINALS:
            return self.value
        if self.value == 'A*':
            return f"A*({self.left})"
        return f"({self.left} {self.value} {self.right})"

def random_tree(depth):
    if depth == 0 or (random.random() < 0.5 and depth < MAX_DEPTH):
        return Node(random.choice(TERMINALS))

    op = random.choice(OPERATORS)
    if op == 'A*':
        return Node(op, left=random_tree(depth - 1))
    else:
        return Node(op, left=random_tree(depth - 1), right=random_tree(depth - 1))

def evaluate(node, A, b, x):
    if node.value == 'x':
        return x
    if node.value == 'b':
        return b
    if node.value == 'A*':
        return A @ evaluate(node.left, A, b, x)

    left_val = evaluate(node.left, A, b, x)
    right_val = evaluate(node.right, A, b, x)

    if node.value == '+':
        return left_val + right_val
    if node.value == '-':
        return left_val - right_val
    if node.value == '*':
        # Element-wise multiplication for vectors
        return left_val * right_val
    raise ValueError(f"Unknown operator: {node.value}")

def get_subtree(node, max_depth, depth=0):
    if depth == max_depth or (not node.left and not node.right):
        return node

    possible_nodes = []
    if node.left:
        possible_nodes.append(node.left)
    if node.right:
        possible_nodes.append(node.right)

    chosen_node = random.choice(possible_nodes)
    return get_subtree(chosen_node, max_depth, depth+1)


def copy_tree(node):
    if node is None:
        return None
    return Node(node.value, copy_tree(node.left), copy_tree(node.right))

def crossover(parent1, parent2):
    child1 = copy_tree(parent1)
    child2 = copy_tree(parent2)

    crossover_point1 = get_random_node(child1)
    crossover_point2 = get_random_node(child2)

    # Swap subtrees
    crossover_point1.value, crossover_point2.value = crossover_point2.value, crossover_point1.value
    crossover_point1.left, crossover_point2.left = crossover_point2.left, crossover_point1.left
    crossover_point1.right, crossover_point2.right = crossover_point2.right, crossover_point1.right

    return child1, child2

def get_random_node(node, parent=None, nodes=None):
    if nodes is None:
        nodes = []
    if node is not None:
        nodes.append(node)
        get_random_node(node.left, node, nodes)
        get_random_node(node.right, node, nodes)
    return random.choice(nodes)


def mutate(tree):
    if random.random() < MUTATION_RATE:
        node_to_mutate = get_random_node(tree)
        node_to_mutate.value = random.choice(OPERATORS + TERMINALS)
        if node_to_mutate.value in TERMINALS:
            node_to_mutate.left = None
            node_to_mutate.right = None
        elif node_to_mutate.value == 'A*':
            if node_to_mutate.left is None:
                node_to_mutate.left = random_tree(0)
            node_to_mutate.right = None
        else:
            if node_to_mutate.left is None:
                node_to_mutate.left = random_tree(0)
            if node_to_mutate.right is None:
                node_to_mutate.right = random_tree(0)
    return tree

def tournament_selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        best_in_tournament = None
        best_fitness = -np.inf
        for _ in range(TOURNAMENT_SIZE):
            idx = random.randint(0, len(population) - 1)
            if fitnesses[idx] > best_fitness:
                best_fitness = fitnesses[idx]
                best_in_tournament = population[idx]
        selected.append(best_in_tournament)
    return selected

def generate_problem(n=10):
    A = np.random.rand(n, n)
    A = np.dot(A, A.T)  # ensure SPD
    b = np.random.rand(n)
    eigvals, eigvecs = np.linalg.eigh(A)
    sqrt_eigvals = np.sqrt(eigvals)
    A_sqrt = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    return A, b, A_sqrt

def fitness(individual, num_problems=5):
    total_error = 0
    for _ in range(num_problems):
        A, b, A_sqrt = generate_problem()
        x = np.zeros_like(b)
        try:
            for _ in range(MAX_ITERATIONS):
                # The GP evolves the UPDATE step, making it an additive update
                update = evaluate(individual, A, b, x)
                if np.isnan(update).any() or np.isinf(update).any():
                    total_error += 1e6
                    break
                x = x + update
                if np.isnan(x).any() or np.isinf(x).any():
                    total_error += 1e6
                    break

            # Fitness is the final relative residual norm of the original problem
            norm_b = np.linalg.norm(b)
            if norm_b < 1e-10: # Avoid division by zero for near-zero vector
                norm_b = 1.0
            error = np.linalg.norm(A_sqrt @ x - b) / norm_b
            if np.isnan(error) or np.isinf(error):
                error = 1e6
            total_error += error
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            total_error += 1e6

    return -total_error / num_problems

def main():
    population = [random_tree(MAX_DEPTH) for _ in range(POPULATION_SIZE)]

    best_overall_solver = None
    best_overall_fitness = -np.inf

    for gen in range(MAX_GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]

        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_overall_fitness:
            best_overall_fitness = fitnesses[current_best_idx]
            best_overall_solver = copy_tree(population[current_best_idx])

        print(f"Generation {gen}: Best Fitness = {best_overall_fitness:.4f}, Solver: x_k+1 = x_k + {best_overall_solver}")

        selected_population = tournament_selection(population, fitnesses)

        # Elitism: the best solver survives to the next generation
        next_population = [best_overall_solver]

        # Fill the rest of the population
        while len(next_population) < POPULATION_SIZE:
            p1, p2 = random.sample(selected_population, 2)
            if random.random() < CROSSOVER_RATE:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1, p2

            next_population.append(mutate(c1))
            if len(next_population) < POPULATION_SIZE:
                next_population.append(mutate(c2))

        population = next_population

    print(f"\nBest solver found: x_k+1 = x_k + {best_overall_solver}")

    with open('gp_sqrt_solver_experiment/best_solver.pkl', 'wb') as f:
        pickle.dump(best_overall_solver, f)
    print("Best solver saved to gp_sqrt_solver_experiment/best_solver.pkl")

if __name__ == "__main__":
    main()
