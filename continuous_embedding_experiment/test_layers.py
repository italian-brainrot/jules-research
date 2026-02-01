import torch
from continuous_embedding_experiment.layers import FourierFeatures, LinearInterpolationEmbedding, DACE

def test_fourier_features():
    input_dim = 10
    output_dim = 80
    layer = FourierFeatures(input_dim, output_dim)
    x = torch.randn(5, input_dim)
    out = layer(x)
    assert out.shape == (5, output_dim)

def test_linear_interpolation_embedding():
    input_dim = 10
    num_embeddings = 32
    embedding_dim = 16
    layer = LinearInterpolationEmbedding(input_dim, num_embeddings, embedding_dim)
    x = torch.randn(5, input_dim)
    out = layer(x)
    assert out.shape == (5, input_dim * embedding_dim)

def test_dace():
    input_dim = 10
    num_embeddings = 32
    embedding_dim = 16
    layer = DACE(input_dim, num_embeddings, embedding_dim)
    x = torch.randn(5, input_dim)
    out = layer(x)
    assert out.shape == (5, input_dim * embedding_dim)

    loss = layer.get_smoothness_loss()
    assert loss.item() >= 0

if __name__ == "__main__":
    test_fourier_features()
    test_linear_interpolation_embedding()
    test_dace()
    print("All layer tests passed!")
