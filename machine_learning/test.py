# Write your test function in this script.
from train import get_dataset, validate
import torch
from torch import nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    model_version = "2_0"

    _, test_dataset, _ = get_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model = torch.load(f"machine_learning/model/{model_version}.pth")
    device = torch.device("mps" if torch.cuda.is_available() else "cpu") # replacd mps by cuda when using Nvidia
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    test_loss = validate(model, test_dataloader, criterion, device)

    print(f"The MSE of the model on test data is {test_loss}.")
