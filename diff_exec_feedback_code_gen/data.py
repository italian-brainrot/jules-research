import random
import torch
import os
import json

def generate_rpn():
    vars = ['a', 'b', 'c']
    consts = [str(i) for i in range(10)]
    operands = vars + consts
    ops = ['+', '-', '*']

    expressions = []

    # Length 3: var var op
    for v1 in operands:
        for v2 in operands:
            for op in ops:
                expr = [v1, v2, op]
                expressions.append(expr)

    # Length 5:
    # Type 1: var var op var op  (e.g. a b + c *)
    for v1 in operands:
        for v2 in operands:
            for op1 in ops:
                for v3 in operands:
                    for op2 in ops:
                        expr = [v1, v2, op1, v3, op2]
                        expressions.append(expr)

    # Type 2: var var var op op (e.g. a b c * +)
    for v1 in operands:
        for v2 in operands:
            for v3 in operands:
                for op1 in ops:
                    for op2 in ops:
                        expr = [v1, v2, v3, op1, op2]
                        expressions.append(expr)

    return expressions

def evaluate_rpn(expr, inputs):
    stack = []
    for token in expr:
        if token in ['a', 'b', 'c']:
            stack.append(inputs[token])
        elif token.isdigit():
            stack.append(float(token))
        else:
            if len(stack) < 2: return None # Invalid
            v2 = stack.pop()
            v1 = stack.pop()
            if token == '+': stack.append(v1 + v2)
            elif token == '-': stack.append(v1 - v2)
            elif token == '*': stack.append(v1 * v2)
    return stack[0] if len(stack) == 1 else None

def get_description(expr):
    stack = []
    for token in expr:
        if token in ['a', 'b', 'c'] or token.isdigit():
            stack.append(token)
        else:
            v2 = stack.pop()
            v1 = stack.pop()
            if token == '+':
                desc = f"add {v1} and {v2}"
            elif token == '-':
                desc = f"subtract {v2} from {v1}"
            elif token == '*':
                desc = f"multiply {v1} and {v2}"
            stack.append(f"({desc})")
    return stack[0].strip("()")

def create_dataset(num_samples=10000):
    all_exprs = generate_rpn()
    print(f"Total possible expressions: {len(all_exprs)}")

    random.seed(42)
    selected_exprs = random.sample(all_exprs, min(num_samples, len(all_exprs)))

    dataset = []
    for expr in selected_exprs:
        desc = get_description(expr)

        # Generate 5 I/O pairs
        io_pairs = []
        for _ in range(5):
            inputs = {
                'a': float(random.randint(0, 10)),
                'b': float(random.randint(0, 10)),
                'c': float(random.randint(0, 10))
            }
            output = evaluate_rpn(expr, inputs)
            if output is not None:
                io_pairs.append({'inputs': inputs, 'output': float(output)})

        if len(io_pairs) == 5:
            dataset.append({
                'expression': expr,
                'description': desc,
                'io_pairs': io_pairs
            })

    return dataset

if __name__ == "__main__":
    dataset = create_dataset(10000)
    print(f"Final dataset size: {len(dataset)}")

    # Split
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]

    with open("diff_exec_feedback_code_gen/train.json", "w") as f:
        json.dump(train_data, f)
    with open("diff_exec_feedback_code_gen/test.json", "w") as f:
        json.dump(test_data, f)

    # Vocab
    expr_tokens = ['a', 'b', 'c', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*']
    desc_tokens = ['add', 'and', 'subtract', 'from', 'multiply', 'then'] + expr_tokens

    vocab = sorted(list(set(desc_tokens)))
    token_to_idx = {t: i+4 for i, t in enumerate(vocab)}
    token_to_idx['<PAD>'] = 0
    token_to_idx['<SOS>'] = 1
    token_to_idx['<EOS>'] = 2
    token_to_idx['<UNK>'] = 3

    with open("diff_exec_feedback_code_gen/vocab.json", "w") as f:
        json.dump(token_to_idx, f)
    print("Vocab and datasets saved.")
