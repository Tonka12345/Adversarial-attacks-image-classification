from sgd import train, imageClassifier, trainloader, testloader, criterion, optimizer, epochs

if __name__ == "__main__":
    train(imageClassifier, trainloader, testloader, criterion, optimizer, epochs)
