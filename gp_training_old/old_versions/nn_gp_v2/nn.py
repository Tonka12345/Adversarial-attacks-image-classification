import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Definiramo neuronsku mrežu
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.hidden = nn.Linear(784, 512)
        self.output = nn.Linear(512, 10)    # Međusloj
        self.softmax = nn.LogSoftmax(dim=1)
    
    
    def forward(self, x):
        # Prvi sloj: množenje s težinama i dodavanje pomaka
        x = self.hidden(x)
        # Aktivacijska funkcija ReLU (zamjenjuje negativne vrijednosti nulom)
        #x = torch.relu(x)
        #x = self.dropout1(x)
        # Drugi sloj: množenje s težinama i dodavanje pomaka
        x = self.output(x)
        x = self.softmax(x)
        
        return x

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./', train = True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)

