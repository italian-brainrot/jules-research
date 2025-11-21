import torch
import random
import numpy as np
from copy import deepcopy

# Define the functions and terminals
FUNCTIONS = ['add', 'sub', 'mul', 'pdiv', 'sqrt', 'square', 'neg']
TERMINALS = ['param', 'grad', 'one', 'zero', 'momentum', 'learning_rate']

class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def __repr__(self):
        return f"Node({self.value})"

def create_random_tree(max_depth, current_depth=0):
    if current_depth == max_depth or random.random() < 0.3:
        # Choose a terminal
        value = random.choice(TERMINALS)
        return Node(value)
    else:
        # Choose a function
        value = random.choice(FUNCTIONS)
        if value in ['add', 'sub', 'mul', 'pdiv']:
            children = [create_random_tree(max_depth, current_depth + 1) for _ in range(2)]
        elif value in ['sqrt', 'square', 'neg']:
            children = [create_random_tree(max_depth, current_depth + 1)]
        else:
            children = []
        return Node(value, children)

class GeneticProgrammingOptimizer(torch.optim.Optimizer):
    def __init__(self, params, gp_tree, lr=0.01):
        defaults = dict(lr=lr)
        super(GeneticProgrammingOptimizer, self).__init__(params, defaults)
        self.gp_tree = gp_tree
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['momentum'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data
                momentum = self.state[p]['momentum']
                lr = group['lr']

                # Evaluate the GP tree to get the update rule
                update_value = self._evaluate_tree(self.gp_tree, param, grad, momentum, lr)

                # Update momentum
                self.state[p]['momentum'] = 0.9 * momentum + 0.1 * grad

                # Apply the update
                p.data.add_(-lr * update_value)

        return loss

    def _evaluate_tree(self, node, param, grad, momentum, lr):
        if node.value in TERMINALS:
            if node.value == 'param':
                return param
            elif node.value == 'grad':
                return grad
            elif node.value == 'one':
                return torch.ones_like(param)
            elif node.value == 'zero':
                return torch.zeros_like(param)
            elif node.value == 'momentum':
                return momentum
            elif node.value == 'learning_rate':
                return torch.full_like(param, lr)
        elif node.value in FUNCTIONS:
            child_values = [self._evaluate_tree(child, param, grad, momentum, lr) for child in node.children]
            if node.value == 'add':
                return child_values[0] + child_values[1]
            elif node.value == 'sub':
                return child_values[0] - child_values[1]
            elif node.value == 'mul':
                return child_values[0] * child_values[1]
            elif node.value == 'pdiv':
                # Protected division
                return child_values[0] / (child_values[1] + 1e-6)
            elif node.value == 'sqrt':
                return torch.sqrt(torch.abs(child_values[0]))
            elif node.value == 'square':
                return child_values[0] ** 2
            elif node.value == 'neg':
                return -child_values[0]

from model import LogisticRegression

def initialize_population(population_size, max_depth):
    return [create_random_tree(max_depth) for _ in range(population_size)]

def evaluate_fitness(population, X, y, input_dim):
    fitnesses = []
    for tree in population:
        model = LogisticRegression(input_dim)
        optimizer = GeneticProgrammingOptimizer(model.parameters(), tree)
        criterion = torch.nn.BCELoss()

        # Convert data to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Training loop
        try:
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError("Loss is NaN or Inf")
                loss.backward()
                optimizer.step()
            fitnesses.append(loss.item())
        except (RuntimeError, ValueError):
            fitnesses.append(1e6)  # Penalize individuals that cause errors
    return fitnesses

def selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitnesses)]
        selected.append(population[winner_index])
    return selected

def get_all_nodes(tree):
    nodes = [tree]
    for child in tree.children:
        nodes.extend(get_all_nodes(child))
    return nodes

def crossover(parent1, parent2):
    child1, child2 = deepcopy(parent1), deepcopy(parent2)

    # Get all nodes from each parent
    nodes1 = get_all_nodes(child1)
    nodes2 = get_all_nodes(child2)

    # Select a random node from each parent
    node1 = random.choice(nodes1)
    node2 = random.choice(nodes2)

    # Swap the nodes
    node1.value, node2.value = node2.value, node1.value
    node1.children, node2.children = node2.children, node1.children

    return child1, child2

def mutation(tree, max_depth):
    mutated_tree = deepcopy(tree)
    nodes = get_all_nodes(mutated_tree)
    node_to_mutate = random.choice(nodes)

    # Replace the subtree with a new random subtree
    new_value = random.choice(FUNCTIONS + TERMINALS)
    node_to_mutate.value = new_value
    if new_value in ['add', 'sub', 'mul', 'pdiv']:
        node_to_mutate.children = [create_random_tree(max_depth - 1) for _ in range(2)]
    elif new_value in ['sqrt', 'square', 'neg']:
        node_to_mutate.children = [create_random_tree(max_depth - 1)]
    else:
        node_to_mutate.children = []

    return mutated_tree
