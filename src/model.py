import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512) # feature extraction

        self.pool = nn.MaxPool2d(2, 2) # pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # auto dimension
        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)


    def forward(self, x): # forward pass (output)
        x = self.pool(f.relu(self.bn1(self.conv1(x))))
        x = self.pool(f.relu(self.bn2(self.conv2(x))))
        x = self.pool(f.relu(self.bn3(self.conv3(x))))
        x = self.pool(f.relu(self.bn4(self.conv4(x))))
        x = self.pool(f.relu(self.bn5(self.conv5(x))))
        x = self.pool(f.relu(self.bn6(self.conv6(x))))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = f.relu(self.fc1(x)) 
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


        