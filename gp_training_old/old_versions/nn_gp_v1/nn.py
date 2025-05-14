import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Definiramo neuronsku mrežu
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 64)  # Povećan broj neurona
        self.dropout1 = nn.Dropout(0.2)  # dropout
        self.fc2 = nn.Linear(64, 32)    # Međusloj
        self.dropout2 = nn.Dropout(0.2)  # Još jedan dropout
        self.fc3 = nn.Linear(32, 10)    # Izlazni sloj
    
    
    def forward(self, x):
        # Prvi sloj: množenje s težinama i dodavanje pomaka
        x = self.fc1(x)
        # Aktivacijska funkcija ReLU (zamjenjuje negativne vrijednosti nulom)
        x = torch.relu(x)
        x = self.dropout1(x)
        # Drugi sloj: množenje s težinama i dodavanje pomaka
        x = self.fc2(x)
        x = self.dropout2(x)
        #trecisloj
        x = self.fc3(x)
        # Aktivacija Sigmoid - osigurava da su izlazi između 0 i 1
        x = torch.sigmoid(x)
        
        return x

# Priprema podataka (transformacija MNIST-a)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Pretvara sliku u vektor od 784 piksela
])

# Učitavanje punog dataseta
full_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Odabir podskupa dataseta
subset_size = 5000  # Smanjenje dataseta na 5000 slika
subset_indices = torch.randperm(len(full_trainset))[:subset_size]
subset_dataset = torch.utils.data.Subset(full_trainset, subset_indices)

# Kreiranjeloadera s manjom veličinom batcha
trainloader = torch.utils.data.DataLoader(
    subset_dataset, 
    batch_size=16,
    shuffle=True
)
