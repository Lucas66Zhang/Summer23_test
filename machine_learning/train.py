# Write Data loaders, training procedure and validation procedure in this file.
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from model2 import MultiheadAttentionModel

class RectanglesDataset(Dataset):
    def __init__(self, input_array, label_array):
        self.inputs = torch.tensor(input_array, dtype=torch.float32)
        self.labels = torch.tensor(label_array, dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
def get_dataset():
    train_input = []
    train_label = []

    val_input = []
    val_label = []

    test_input = []
    test_label = []

    for i in range(4, 16):
        input_file = os.path.join("machine_learning/dataset/input", f"num_{i}.txt")
        label_file = os.path.join("machine_learning/dataset/label", f"num_{i}.txt")

        input_data = np.loadtxt(input_file)
        label_data = np.loadtxt(label_file)

        input_data = input_data.reshape(-1, i, 5)
        label_data = label_data.reshape(-1, i, 5)[:,:,[0,1,4]]

        input_data = np.pad(input_data, ((0, 0), (0, 15 - i), (0, 0)), mode='constant')
        label_data = np.pad(label_data, ((0, 0), (0, 15 - i), (0, 0)), mode='constant')

        data_len = input_data.shape[0]
        train_len = int(data_len * 0.7)
        val_len = int(0.2 * data_len)

        train_input.append(input_data[range(train_len)])
        val_input.append(input_data[range(train_len, train_len + val_len)])
        test_input.append(input_data[range(train_len + val_len, data_len)])

        train_label.append(label_data[range(train_len)])
        val_label.append(label_data[range(train_len, train_len + val_len)])
        test_label.append(label_data[range(train_len + val_len, data_len)])

    train_input = np.concatenate(train_input, 0)
    train_label = np.concatenate(train_label, 0)

    val_input = np.concatenate(val_input, 0)
    val_label = np.concatenate(val_label, 0)

    test_input = np.concatenate(test_input, 0)
    test_label = np.concatenate(test_label, 0)

    train_dataset = RectanglesDataset(train_input, train_label)
    test_dataset = RectanglesDataset(test_input, test_label)
    val_dataset = RectanglesDataset(val_input, val_label)

    return train_dataset, test_dataset, val_dataset

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples

    return avg_loss

if __name__ == "__main__":
    '''
    train_input = []
    train_label = []

    val_input = []
    val_label = []

    test_input = []
    test_label = []

    for i in range(4, 16):
        input_file = os.path.join("./input", f"num_{i}.txt")
        label_file = os.path.join("./label", f"num_{i}.txt")

        input_data = np.loadtxt(input_file)
        label_data = np.loadtxt(label_file)

        input_data = input_data.reshape(-1, i, 5)
        label_data = label_data.reshape(-1, i, 5)

        input_data = np.pad(input_data, ((0, 0), (0, 15 - i), (0, 0)), mode='constant')
        label_data = np.pad(label_data, ((0, 0), (0, 15 - i), (0, 0)), mode='constant')

        data_len = input_data.shape[0]
        train_len = int(data_len * 0.7)
        val_len = int(0.2 * data_len)

        train_input.append(input_data[range(train_len)])
        val_input.append(input_data[range(train_len, train_len + val_len)])
        test_input.append(input_data[range(train_len + val_len, data_len)])

        train_label.append(label_data[range(train_len)])
        val_label.append(label_data[range(train_len, train_len + val_len)])
        test_label.append(label_data[range(train_len + val_len, data_len)])

    train_input = np.concatenate(train_input, 0)
    train_label = np.concatenate(train_label, 0)

    val_input = np.concatenate(val_input, 0)
    val_label = np.concatenate(val_label, 0)

    test_input = np.concatenate(test_input, 0)
    test_label = np.concatenate(test_label, 0)

    train_dataset = RectanglesDataset(train_input, train_label)
    test_dataset = RectanglesDataset(test_input, test_label)
    val_dataset = RectanglesDataset(val_input, val_label)
    '''
    train_dataset, _, val_dataset = get_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    # test_dataloader = DataLoader(test_dataset, batch_size=1)

    model = MultiheadAttentionModel(input_dim=5, output_dim=3, num_heads=5, num_layers=2)

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 40

    for data in train_dataloader:
        inputs, labels = data
        if torch.isnan(inputs).any() or torch.isnan(labels).any():
            print(f"DataLoader contains NaN values")
            break
    
    for epoch in range(num_epochs):
        train(model, train_dataloader, criterion, optimizer, device)
        train_loss = validate(model, train_dataloader, criterion, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss} | Validation Loss: {val_loss}")

    model_version = "2_0"
    torch.save(model, f"machine_learning/model/{model_version}.pth")
    