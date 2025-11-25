import random
import copy

class Node:
    def __ne__(self, other):
        return not self.__eq__(other)

class Var(Node):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def to_pytorch(self):
        return self.name

class App(Node):
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg

    def __str__(self):
        return f"({self.func} {self.arg})"

    def __eq__(self, other):
        return isinstance(other, App) and self.func == other.func and self.arg == other.arg

    def to_pytorch(self):
        # Unpack nested Apps for multi-argument functions
        args = [self.arg]
        func = self.func
        while isinstance(func, App):
            args.append(func.arg)
            func = func.func
        args.reverse()
        return f"{func.to_pytorch()}({', '.join(arg.to_pytorch() for arg in args)})"

class Lambda(Node):
    def __init__(self, var, body):
        self.var = var
        self.body = body

    def __str__(self):
        return f"(λ{self.var}. {self.body})"

    def __eq__(self, other):
        return isinstance(other, Lambda) and self.var == other.var and self.body == other.body

    def to_pytorch(self):
        return f"(lambda {self.var.to_pytorch()}: {self.body.to_pytorch()})"

class Individual:
    def __init__(self, expression):
        self.expression = expression
        self.fitness = -1

    def __str__(self):
        return str(self.expression)

    def to_pytorch(self):
        return self.expression.to_pytorch()

def random_expression(primitives, terminals, max_depth, depth=0):
    if depth >= max_depth or random.random() < 0.3:
        return Var(random.choice(terminals))
    else:
        chosen_primitive = random.choice(primitives)
        if chosen_primitive == 'lambda':
            var_name = random.choice(['x', 'y', 'z']) # Simple variable names for now
            return Lambda(Var(var_name), random_expression(primitives, terminals, max_depth, depth + 1))
        else:
            op = Var(chosen_primitive)
            arg1 = random_expression(primitives, terminals, max_depth, depth + 1)
            return App(op, arg1)

def subtree_mutation(individual, primitives, terminals, max_depth):
    new_expr = copy.deepcopy(individual.expression)
    nodes = get_all_nodes(new_expr)
    node_to_mutate = random.choice(nodes)

    # To avoid making a shallow copy and modifying the original tree
    new_subtree = random_expression(primitives, terminals, max_depth=max_depth-get_depth(node_to_mutate, new_expr))

    if node_to_mutate is new_expr:
        return Individual(new_subtree)

    parent = find_parent(new_expr, node_to_mutate)
    if isinstance(parent, App):
        if parent.func is node_to_mutate:
            parent.func = new_subtree
        else:
            parent.arg = new_subtree
    elif isinstance(parent, Lambda):
        parent.body = new_subtree

    return Individual(new_expr)


def subtree_crossover(parent1, parent2):
    child1_expr = copy.deepcopy(parent1.expression)
    child2_expr = copy.deepcopy(parent2.expression)

    nodes1 = get_all_nodes(child1_expr)
    node1 = random.choice(nodes1)

    nodes2 = get_all_nodes(child2_expr)
    node2 = random.choice(nodes2)

    parent_of_node1 = find_parent(child1_expr, node1)
    parent_of_node2 = find_parent(child2_expr, node2)

    if parent_of_node1 is None: # root
        child1_expr = node2
    elif isinstance(parent_of_node1, App):
        if parent_of_node1.func is node1:
            parent_of_node1.func = node2
        else:
            parent_of_node1.arg = node2
    elif isinstance(parent_of_node1, Lambda):
        parent_of_node1.body = node2

    if parent_of_node2 is None: # root
        child2_expr = node1
    elif isinstance(parent_of_node2, App):
        if parent_of_node2.func is node1:
            parent_of_node2.func = node1
        else:
            parent_of_node2.arg = node1
    elif isinstance(parent_of_node2, Lambda):
        parent_of_node2.body = node1

    return Individual(child1_expr), Individual(child2_expr)

def get_all_nodes(expr):
    nodes = [expr]
    if isinstance(expr, App):
        nodes.extend(get_all_nodes(expr.func))
        nodes.extend(get_all_nodes(expr.arg))
    elif isinstance(expr, Lambda):
        nodes.extend(get_all_nodes(expr.body))
    return nodes

def find_parent(root, child):
    if root is child:
        return None
    if isinstance(root, App):
        if root.func is child or root.arg is child:
            return root
        parent = find_parent(root.func, child)
        if parent: return parent
        return find_parent(root.arg, child)
    if isinstance(root, Lambda):
        if root.body is child:
            return root
        return find_parent(root.body, child)
    return None

def get_depth(node, root, depth=0):
    if root is node:
        return depth
    if isinstance(root, App):
        d = get_depth(node, root.func, depth + 1)
        if d is not None: return d
        return get_depth(node, root.arg, depth + 1)
    if isinstance(root, Lambda):
        return get_depth(node, root.body, depth + 1)
    return None

# Define the primitives and terminals for the optimizer search space
primitives = ['add', 'sub', 'mul', 'safe_div', 'sqrt', 'neg']
terminals = ['g', 'm', 'v', 'one', 'zero']
arity = {'add': 2, 'sub': 2, 'mul': 2, 'safe_div': 2, 'sqrt': 1, 'neg': 1, 'lambda': 1}

def random_expression(primitives, terminals, max_depth, depth=0):
    if depth >= max_depth or random.random() < 0.3:
        return Var(random.choice(terminals))
    else:
        chosen_primitive = random.choice(primitives)
        if chosen_primitive == 'lambda':
            var_name = random.choice(['x', 'y', 'z']) # Simple variable names for now
            return Lambda(Var(var_name), random_expression(primitives, terminals, max_depth, depth + 1))
        else:
            op = Var(chosen_primitive)
            num_args = arity[chosen_primitive]
            expr = op
            for _ in range(num_args):
                expr = App(expr, random_expression(primitives, terminals, max_depth, depth + 1))
            return expr

def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mul(x, y):
    return x * y

def safe_div(x, y):
    return x / (y + 1e-8)

def sqrt(x):
    return (x.abs() + 1e-8).sqrt()

def neg(x):
    return -x

one = 1.0
zero = 0.0

# Register the functions in the global scope
func_map = {
    'add': add,
    'sub': sub,
    'mul': mul,
    'safe_div': safe_div,
    'sqrt': sqrt,
    'neg': neg,
    'one': one,
    'zero': zero
}

if __name__ == '__main__':
    # Example usage
    # Create a random individual
    individual1 = Individual(random_expression(primitives, terminals, max_depth=3))
    print(f"Individual 1: {individual1}")
    print(f"Pytorch code for Individual 1: {individual1.to_pytorch()}")


    # Create another random individual
    individual2 = Individual(random_expression(primitives, terminals, max_depth=3))
    print(f"Individual 2: {individual2}")

    # Crossover
    child1, child2 = subtree_crossover(individual1, individual2)
    print(f"Child 1 after crossover: {child1}")
    print(f"Child 2 after crossover: {child2}")

    # Mutation
    mutated_individual = subtree_mutation(individual1, primitives, terminals, max_depth=3)
    print(f"Mutated Individual 1: {mutated_individual}")
