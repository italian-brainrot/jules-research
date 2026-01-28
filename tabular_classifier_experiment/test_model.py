import torch
from model import PrototypeClassifier

def test_overfit():
    input_dim = 40
    output_dim = 10
    batch_size = 8
    model = PrototypeClassifier(input_dim, output_dim, n_prototypes_per_class=5)
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(200):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print(f"Iter {i}, Loss: {loss.item()}")

    pred = out.argmax(dim=1)
    print(f"Target: {y}")
    print(f"Pred:   {pred}")

if __name__ == "__main__":
    test_overfit()
