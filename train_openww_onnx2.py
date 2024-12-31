import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Dummy Dataset for Wake Word Detection
class WakeWordDataset(Dataset):
    def __init__(self):
        self.data = np.load('./spectrograms/data.npy')
        self.labels = np.load('./spectrograms/labels.npy')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dim
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# Model Architecture
class WakeWordModel(nn.Module):
    def __init__(self):
        super(WakeWordModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(20608, 128)  # Update based on your flattened size
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        print("Input Shape:", x.shape)
        x = self.pool(torch.relu(self.conv1(x)))
        print("After Conv1 + Pool:", x.shape)
        x = self.pool(torch.relu(self.conv2(x)))
        print("After Conv2 + Pool:", x.shape)
        x = x.view(x.size(0), -1)  # Dynamic Flattening
        print("After Flatten:", x.shape)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Training Loop
dataset = WakeWordDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = WakeWordModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # 10 Epochs
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

# Save Model
torch.save(model.state_dict(), 'wakeword_model.pth')
print("Model trained and saved as wakeword_model.pth")
