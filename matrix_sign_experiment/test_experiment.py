
import numpy as np
from .main import generate_matrix_stream

def test_generate_matrix_stream():
    n_steps = 10
    n_dim = 5
    stream = list(generate_matrix_stream(n_steps=n_steps, n_dim=n_dim))

    assert len(stream) == n_steps
    for A in stream:
        assert A.shape == (n_dim, n_dim)
        assert not np.isnan(A).any()
        assert np.isfinite(A).all()
