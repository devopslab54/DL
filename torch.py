import torch
import torch.nn as nn
import torch.optim as optim

# Data
train_data = torch.rand(1000, 784)
train_labels = torch.randint(0, 10, (1000,))

# Model
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    for i in range(0, len(train_data), 32):
        batch_data = train_data[i:i+32]
        batch_labels = train_labels[i:i+32]
        loss = criterion(model(batch_data), batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Evaluation
accuracy = (model(train_data).argmax(dim=1) == train_labels).float().mean()
print(f"Accuracy: {accuracy:.4f}")
