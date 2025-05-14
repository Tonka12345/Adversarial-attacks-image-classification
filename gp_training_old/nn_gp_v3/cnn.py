import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class CNNImageClassifier(nn.Module):
    def __init__(self):
        super(CNNImageClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        #input [batch_size, channels, height, width]
        if x.dim() == 2:
            # If flattened (784), reshape to [batch_size, 1, 28, 28]
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3 and x.size(1) == 784:
            # If [batch_size, 784], reshape to [batch_size, 1, 28, 28]
            x = x.view(-1, 1, 28, 28)
            
        # First conv layer followed by ReLU activation
        x = F.relu(self.conv1(x))
        
        # Max pooling to reduce spatial dimensions (28x28 -> 14x14)
        x = self.pool(x)
        
        # Second conv layer followed by ReLU activation
        x = F.relu(self.conv2(x))
        
        # Max pooling again (14x14 -> 7x7)
        x = self.pool(x)

        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))        
        x = self.fc2(x)        
        x = self.softmax(x)
        
        return x