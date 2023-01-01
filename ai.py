import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# load the mnist dataset and let a user search for a fashion item. Use a machine learning reinfocement learning algorithm to find the item.

# load the mnist dataset
train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor(), download=True)

# create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# define the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# define the training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# define the testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# define the main function let the user search for a fashion item using a machine learning reinforcement learning algorithm
def main():
    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define the model
    model = Model().to(device)
    
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # define the number of epochs
    epochs = 10
    
    # train the model
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        print('Epoch: {}, Test Loss: {:.4f}, Test Accuracy: {:.2f}'.format(epoch, test_loss, test_accuracy))
    
    # save the model
    torch.save(model.state_dict(), 'model.pth')
    
    # load the model
    model.load_state_dict(torch.load('model.pth'))
    
    # let user search for a fashion item using a machine learning reinforcement learning algorithm. plot the data the model found
    with torch.no_grad():
        print ('Let\'s search for a fashion item using a machine learning reinforcement learning algorithm.')
        print ('Enter a number between 0 and 9 to search for a fashion item.')
        print ('Enter -1 to exit.')
        while True:
            num = int(input('Enter a number: '))
            if num == -1:
                break
            else:
                for batch_idx, (data, target) in enumerate(test_loader):
                    data = data.to(device)
                    target = target.to(device)
                    data = data.view(data.size(0), -1)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    for i in range(len(pred)):
                        if pred[i] == num:
                            plt.imshow(data[i].cpu().numpy().reshape(28, 28), cmap='gray')
                            plt.show()
                            break
# call the main function
if __name__ == '__main__':
    main()

