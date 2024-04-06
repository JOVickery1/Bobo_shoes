import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split, default_collate
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Read data from file
with open('iris.data', 'r') as file:
    lines = file.readlines()

# Parse the data into a list of lists (or any other appropriate data structure)
data = []
labels = []
for i, line in enumerate(lines):
    if i == 150: break
    # Parse each line and convert it to the appropriate data type
    parts = line.strip().split(',')
    # Assuming the first four columns represent features and the last column represents labels
    features = [float(x) for x in parts[:-1]]
    if len(features) != 4:
            print("Invalid data entry:", parts)
            print("Invalid line: ", i)
    label = parts[-1]
    data.append(features)
    labels.append(label)

# Convert labels to numerical values
label_to_index = {label: index for index, label in enumerate(set(labels))}
numeric_labels = [label_to_index[label] for label in labels]

class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.Y = labels

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, ndata in enumerate(train_loader, 0):
      inputs, labels = ndata

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Add the current loss to the running loss
      running_loss += loss.item()

    # Average over the Epoch
    running_loss = running_loss/len(train_loader)
    return running_loss

def test(model, test_loader):
    model.eval

    test_loss = 0

    # enumerate over the data
    for i, ndata in enumerate(test_loader,0):
        inputs, labels = ndata
        # Forward pass
        test_outputs = model(inputs)
        # Calculate loss with criterion
        loss = criterion(test_outputs, labels)

        # Add the current loss to the running loss
        test_loss += loss.item()
    
    # Average test lsos over the epoch
    test_loss = test_loss/len(test_loader)
    return test_loss

# Convert the data and labels lists to tensors

tensor_data = torch.tensor(data).to(device)
tensor_labels = torch.tensor(numeric_labels).to(device)

# tensor_data = torch.tensor(data)
# tensor_labels = torch.tensor(numeric_labels)

print("-------------")
print(f"On CUDA: {tensor_data.is_cuda}")
print("-------------")

dataset = MyDataset(tensor_data, tensor_labels)

model = Net(tensor_data.shape[1])
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 64

split = 0.2
test_size = int(np.floor(split * len(dataset)))
train_size = len(dataset) - test_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)

epochs = 100

start_time = time.time()

for epoch in range(epochs):
    # Training pass
    loss = train(model, train_loader, optimizer, criterion)
    # Testing pass
    test_loss = test(model, test_loader)

    # Print stats every 10 Epochs
    if(epoch % 10 == 0):
        print(f'Training Epoch [{epoch}/{epochs}], Train Loss: {loss:.7f}, Test Loss: {test_loss:.7f}')

print(f"--- {time.time()-start_time} seconds ---")