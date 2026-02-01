import torch
from layers import HarmonicParameterizedEmbedding, FactorizedEmbedding, StandardEmbedding

def test_layers():
    V, D, K = 100, 64, 8
    batch, seq_len = 16, 20
    x = torch.randint(0, V, (batch, seq_len))

    models = [
        HarmonicParameterizedEmbedding(V, D, K),
        FactorizedEmbedding(V, D, K),
        StandardEmbedding(V, D)
    ]

    for model in models:
        out = model(x)
        assert out.shape == (batch, seq_len, D), f"{type(model)} failed shape check: {out.shape}"
        loss = out.sum()
        loss.backward()
        print(f"{type(model)} passed.")

if __name__ == "__main__":
    test_layers()
