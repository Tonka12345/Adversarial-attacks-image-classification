import torch, torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
import os
from display_learning import display_learning


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(28*28, 512)
        self.hidden2 = nn.Linear(512, 64)
        self.output = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x
    
imageClassifier = ImageClassifier()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]) #normaliziramo vrijednosti pixela

trainset = torchvision.datasets.MNIST(root='./', train = True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

criterion = nn.NLLLoss()

optimizer = optim.SGD(imageClassifier.parameters(), lr=0.01)

epochs = 10

def save_model(model, filename):
    model_path = f"{filename}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model spremljen u {model_path}")

def load_model(model, filename):
    model_path = f"{filename}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model učitan iz {model_path}")
        return True
    else:
        print(f"Model {model_path} nije pronađen")
        return False
    
def calculate_accuracy(imageClassifier, loader):
    imageClassifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.shape[0], -1)
            outputs = imageClassifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()    
    accuracy = 100 * correct / total
    return accuracy

def calculate_loss(imageClassifier, loader, criterion):
    #breakpoint()
    imageClassifier.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.shape[0], -1)
            probs = imageClassifier(images)
            total_loss += criterion(probs, labels)
            total += 1
    loss = total_loss/total
    return loss

def train(imageClassifier, trainloader, testloader, criterion, optimizer, epochs):
    accuracies_test = []
    accuracies_train = []
    losses_test=[]
    losses_train=[]

    for epoch in range(epochs):
        #ucimo na skupu podataka za treniranje
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)  # Flatten images
            optimizer.zero_grad()  # Zero the gradients
            probs = imageClassifier(images)  # Forward pass
            loss = criterion(probs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() #nije dobro koristiti kao mjeru jer se model mijenja tijekom epohe(bolje na kraju izracunati)

        #nakon svake epohe evaluiramo na testnom skupu podataka    
        accuracy_test = calculate_accuracy(imageClassifier, testloader)

        #jos podataka za grafove
        accuracies_test.append(accuracy_test)
        accuracies_train.append(calculate_accuracy(imageClassifier, trainloader))
        #losses_train.append(running_loss/len(trainloader))
        losses_test.append(calculate_loss(imageClassifier, testloader, criterion))
        losses_train.append(calculate_loss(imageClassifier, trainloader, criterion))


        print(f"Epoch {epoch+1}/{epochs} - Loss: {losses_train[epoch]:.4f}, Accuracy: {accuracy_test:.2f}%")

    save_model(imageClassifier, "last_trained")
    display_learning(epochs, accuracies_test, accuracies_train, losses_test, losses_train)

#train(imageClassifier, trainloader, testloader, criterion, optimizer, epochs)
