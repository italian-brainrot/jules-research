import torch
from soft_tokenization_experiment.compare import get_data
from soft_tokenization_experiment.model import TokenizationModel

def test_text_loading():
    print("Testing data loading...")
    (X_train, y_train), (X_test, y_test) = get_data(seq_len=128)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    assert X_train.shape[1] == 128
    assert y_train.dim() == 1

    print("Testing model with text data...")
    model = TokenizationModel(vocab_size=256, input_len=128, output_len=32, dim=64, nhead=4, num_layers=2, method="BGST")
    logits = model(X_train[:8])
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (8, 4)

    loss = logits.sum()
    loss.backward()
    print("Backward pass successful.")

if __name__ == "__main__":
    test_text_loading()
