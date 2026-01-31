import torch
import torch.nn.functional as F
from model_gen import Generator
from executor import ProxyExecutor
from compare import evaluate_rpn

def test_evaluate_rpn():
    assert evaluate_rpn(['1', '2', '+'], {'a':0, 'b':0, 'c':0}) == 3.0
    assert evaluate_rpn(['a', 'b', '*'], {'a':2, 'b':3, 'c':0}) == 6.0
    assert evaluate_rpn(['a', 'b', '+', 'c', '*'], {'a':1, 'b':2, 'c':3}) == 9.0
    assert evaluate_rpn(['a', 'b', 'c', '*', '+'], {'a':1, 'b':2, 'c':3}) == 7.0

def test_generator_forward():
    vocab_size = 20
    model = Generator(vocab_size, 16, 32)
    x = torch.randint(0, vocab_size, (4, 8))
    out = model(x, target_len=5)
    assert out.shape == (4, 5, vocab_size)

def test_generator_forward_soft():
    vocab_size = 20
    model = Generator(vocab_size, 16, 32)
    x = torch.randint(0, vocab_size, (4, 8))
    out = model.forward_soft(x, target_len=5)
    assert out.shape == (4, 5, vocab_size)
    assert out.requires_grad

def test_executor_forward():
    vocab_size = 20
    model = ProxyExecutor(vocab_size, 16, 32)
    # Hard tokens
    x_hard = torch.randint(0, vocab_size, (4, 5))
    inputs = torch.randn(4, 3)
    out_hard = model(x_hard, inputs)
    assert out_hard.shape == (4,)

    # Soft tokens
    x_soft = torch.randn(4, 5, vocab_size).softmax(dim=-1)
    out_soft = model(x_soft, inputs)
    assert out_soft.shape == (4,)
