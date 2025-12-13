
import random
import math

# Define the function set
def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y): return x / y if y != 0 else 1.0
def sin(x): return math.sin(x)
def cos(x): return math.cos(x)
def exp(x): return math.exp(x) if x < 700 else float('inf')
def log(x): return math.log(abs(x)) if x != 0 else 0.0


FUNCTIONS = [add, sub, mul, div, sin, cos, exp, log]
FUNCTION_NAMES = {f.__name__: f for f in FUNCTIONS}
FUNCTION_ARITIES = {
    add: 2, sub: 2, mul: 2, div: 2,
    sin: 1, cos: 1, exp: 1, log: 1
}

# Define the terminal set
TERMINALS = ['epoch', 'loss', 'val_loss']

class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def __str__(self):
        if not self.children:
            return str(self.value)
        if len(self.children) == 1:
             return f"{self.value.__name__}({self.children[0]})"
        return f"({self.children[0]} {self.value.__name__} {self.children[1]})"

def create_random_expression(max_depth=3):
    if max_depth == 0 or random.random() < 0.3:
        # Terminal node
        terminal = random.choice(TERMINALS + [random.uniform(-1, 1)])
        return Node(terminal)
    else:
        # Function node
        func = random.choice(FUNCTIONS)
        arity = FUNCTION_ARITIES[func]
        children = [create_random_expression(max_depth - 1) for _ in range(arity)]
        return Node(func, children)

def evaluate_expression(node, **bindings):
    if isinstance(node.value, (float, int)):
        return node.value
    if isinstance(node.value, str):
        return bindings.get(node.value, 0.0) # Default to 0 if not found

    # It's a function
    func = node.value
    arity = FUNCTION_ARITIES[func]
    args = [evaluate_expression(child, **bindings) for child in node.children]

    # Handle arity mismatch gracefully, though our creation logic should prevent this
    if len(args) != arity:
        return 1.0 # Return a default value

    try:
        result = func(*args)
        # Handle potential numerical instability from functions like exp
        if not math.isfinite(result):
            return 1.0  # Return a large but finite number
        return result
    except (ValueError, OverflowError):
        return 1.0 # Return a default/safe value on math errors

def crossover(parent1, parent2):
    child1, child2 = clone_tree(parent1), clone_tree(parent2)

    nodes1, nodes2 = get_all_nodes(child1), get_all_nodes(child2)

    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2)

    # Simple swap of the whole node. A more robust implementation might
    # check for type compatibility or depth limits.
    crossover_point1.value, crossover_point2.value = crossover_point2.value, crossover_point1.value
    crossover_point1.children, crossover_point2.children = crossover_point2.children, crossover_point1.children

    return child1, child2

def mutation(tree, max_depth=3):
    mutated_tree = clone_tree(tree)
    nodes = get_all_nodes(mutated_tree)
    mutation_point = random.choice(nodes)

    # Calculate the depth of the mutation point within the tree
    depth_of_mutation_point = get_depth_of_node(mutated_tree, mutation_point)
    if depth_of_mutation_point is None:
        depth_of_mutation_point = 1 # Should not happen if logic is correct

    # The max_depth for the new subtree should be constrained by the overall max_depth
    new_subtree_max_depth = max_depth - (depth_of_mutation_point - 1)

    if new_subtree_max_depth < 1:
        # If we are already at max depth, the new "subtree" must be a terminal
        new_subtree = create_random_expression(max_depth=0)
    else:
        new_subtree = create_random_expression(max_depth=new_subtree_max_depth)

    mutation_point.value = new_subtree.value
    mutation_point.children = new_subtree.children

    return mutated_tree

def clone_tree(node):
    if node is None:
        return None
    cloned_children = [clone_tree(child) for child in node.children]
    return Node(node.value, cloned_children)

def get_all_nodes(node):
    nodes = [node]
    for child in node.children:
        nodes.extend(get_all_nodes(child))
    return nodes

def get_depth(node):
    # This function calculates the height of the tree/subtree starting from 'node'
    if not node or not node.children:
        return 1
    return 1 + max(get_depth(child) for child in node.children)

def get_depth_of_node(root, target_node, current_depth=1):
    # This function finds the depth of a specific 'target_node' within the tree starting from 'root'
    if root == target_node:
        return current_depth
    for child in root.children:
        depth = get_depth_of_node(child, target_node, current_depth + 1)
        if depth is not None:
            return depth
    return None
