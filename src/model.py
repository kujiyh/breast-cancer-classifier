import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=1) # feature extraction

        self.pool = nn.MaxPool2d(2, 2) # pooling

        self.fc1 = nn.Linear(64 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 2) # the great flattening


    def forward(self, x): # forward pass (output)
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x))) 
        x = self.pool(f.relu(self.conv3(x)))

        x = x.view(-1, 64 * 26 * 26)

        x = f.relu(self.fc1(x)) 
        x = self.fc2(x) # flatten again
        
        return x


        