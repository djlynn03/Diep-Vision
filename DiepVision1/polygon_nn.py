import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import polygon_dataset

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# train_dataloader = DataLoader(polygon_dataset.PolygonImageDataset(annotations_file='.\\images\\labels.csv', img_dir='.\\images'), batch_size=64, shuffle=True)
# print("Train: ", len(train_dataloader.dataset))
# test_dataloader = DataLoader(polygon_dataset.PolygonImageDataset(annotations_file='.\\test_images\\labels.csv', img_dir='.\\test_images'), batch_size=64, shuffle=True)
# print("Test: ", len(test_dataloader.dataset))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12288, 4096),
            nn.ReLU(),
            nn.Linear(4096,512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x.float())
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f} | Batch: {current}/{size}")
            
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
        
# model = NeuralNetwork().to(device)


# learning_rate = 1e-2
# batch_size = 3
# epochs = 5

# loss_fn = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# epochs = 100
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")


        
