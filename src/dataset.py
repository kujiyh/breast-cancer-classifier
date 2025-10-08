import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

def get_loaders(data_dir = 'C:/Users/brian/BCC/data', batch_size=32): # load data
    transform = transforms.Compose([transforms.Resize((224, 224)),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # converting to tensor and standardizing data
    ])

    dataset = datasets.ImageFolder(data_dir, transform = transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size 
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) # split data for training and testing

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) # nonrandom test data

    return train_loader, test_loader, dataset.classes




