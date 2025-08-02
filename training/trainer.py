import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, criterion, optimizer, device, epochs=1):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {running_loss / len(train_loader):.4f}")
