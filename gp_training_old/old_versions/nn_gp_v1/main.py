from nn import ImageClassifier, trainloader
from evolution import genetic_algorithm
import torch

def main():
    # Kreiramo populaciju modela
    population_size = 100
    models = [ImageClassifier() for _ in range(population_size)]

    # Pokrećemo evolucijsko učenje
    best_model = genetic_algorithm(models, trainloader, generations=10000, mutation_rate=0.7)

    # Spremamo najbolji model
    torch.save(best_model.state_dict(), "best_model.pth")
    print("Najbolji model spremljen kao 'best_model.pth'.")

if __name__ == "__main__":
    main()