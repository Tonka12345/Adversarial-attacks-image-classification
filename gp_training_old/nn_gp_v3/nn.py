import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.hidden = nn.Linear(784, 512)
        self.output = nn.Linear(512, 10)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root='./', train = True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


