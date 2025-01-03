from CNN import CNN
from load_data import load_data
import torch.optim as optim
import torch.nn as nn
from typing import List
import torch
from torch.utils.data import DataLoader

def train_model(num_epochs=5) -> CNN:
    device = torch.device("mps")
    trainloader, testloader = load_data()
    print("loaded data")

    model: CNN = CNN().to(device)
    optimizer: torch.optim.AdamW = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    loss_list: List[float] = []

    for _ in range(num_epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_val = loss_fn(outputs, labels)
            loss_val.backward()
            optimizer.step()
            loss_list.append(loss_val.item())

            if i % 100 == 0:
                print(f"At batch {i} in epoch {_} the training loss is {loss_val.item()}")
                print(f"At batch {i} in epoch {_} the test accuracy is {evaluate_on_test_set(testloader, model)}")

    print("finished training")
    torch.save(model, 'CNN.pth')
    print("saved the model")
    return model

def evaluate_on_test_set(testloader: DataLoader, model: CNN) -> float:
    device = torch.device("mps")
    model.eval()
    correct: float = 0.0
    total: float = 0.0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.max(outputs, dim=-1).indices
            total += labels.shape[0]
            correct += (predictions == labels).sum().item()

    return correct/total

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        train_model()
    else:
        print("MPS device not found.")
