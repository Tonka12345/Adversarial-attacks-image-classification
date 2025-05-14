from nn import ImageClassifier, trainloader, testloader
from cnn import CNNImageClassifier
from evolution import genetic_algorithm, genetic_algorithm_generational
import torch

def main():
    population_size = 100

    best_model = genetic_algorithm(population_size, trainloader, testloader, generations=10000, 
                                                 mutation_rate=0.05, model_class=ImageClassifier)

    #best_model = genetic_algorithm_generational(population_size, trainloader, testloader, generations=100, mutation_rate=0.05, model_class=ImageClassifier)

    # Spremamo najbolji model
    torch.save(best_model.state_dict(), "best_model.pth")
    print("Najbolji model spremljen kao 'best_model.pth'.")

if __name__ == "__main__":
    main()