import torch
import random
import copy
import heapq
import numpy
from nn import ImageClassifier, trainloader
from cnn import CNNImageClassifier

def get_chromosome_from_model(model): #vracat ce numpy array
    chromosome = []
    for param in model.parameters():
        flattened = param.data.flatten().tolist()
        chromosome.extend(flattened)
    return numpy.array(chromosome)

def get_model_from_chromosome(chromosome, model_class=ImageClassifier):
    model = model_class()    
    param_shapes = [p.data.shape for p in model.parameters()]
    
    start_idx = 0
    for i, param in enumerate(model.parameters()):
        param_length = param.numel()        
        param_values = chromosome[start_idx:start_idx + param_length]
        param.data = torch.tensor(param_values).float().reshape(param_shapes[i])        
        start_idx += param_length
    return model


def mutate(chromosome, mutation_rate=0.1, mutation_strength=0.1):
    mask = numpy.random.rand(*chromosome.shape) < mutation_rate
    mutations = numpy.random.uniform(-mutation_strength, mutation_strength, chromosome.shape)
    chromosome[mask] += mutations[mask]
    return chromosome

def crossover(p1, p2, method="single_point"):
    size = len(p1)
    c = numpy.copy(p1)

    if method == "single_point":
        point = numpy.random.randint(1, size)
        c[point:] = p2[point:]

    elif method == "two_point":
        point1, point2 = sorted(numpy.random.randint(1, size, 2))
        c[point1:point2] = p2[point1:point2]
    elif method == "uniform":
        mask = numpy.random.rand(size) < 0.5
        c[mask] = p2[mask]
    return c

def evaluate(model, dataloader):
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.NLLLoss() #loss = -log(p(x))
    total_loss = 0.0
    num = 0
    with torch.no_grad():
        for images, labels in dataloader:
            if isinstance(model, CNNImageClassifier):
                outputs = model(images)
            else:
                images = images.view(images.shape[0], -1)
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num+=1
    return num/total_loss  # Manja greška = bolji fitness

def tournament_selection(fitness_scores, k=3):
    tournament = random.sample(fitness_scores, k)  # Nasumično uzimamo k jedinki
    return max(tournament, key=lambda x: x[0])[2] 

def accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            if isinstance(model, CNNImageClassifier):
                outputs = model(images)
            else:
                images = images.view(images.shape[0], -1)
                outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def initialize_population(population_size, model_class):
    models = [model_class() for _ in range(population_size)]
    return models

#steady state gp
def genetic_algorithm(population_size, trainloader, testloader, model_class,
                       generations=10, mutation_rate=0.1):
    
    models = initialize_population(population_size, model_class)
    #dodajemo random u trojku da bi se moglo usporedivati za heapq
    fitness_scores = [(evaluate(model,trainloader), random.random(), model) for model in models]
    heapq.heapify(fitness_scores)
    gen = 0
    a = 0
    while gen < generations:
        p1 = get_chromosome_from_model(tournament_selection(fitness_scores))
        p2 = get_chromosome_from_model(tournament_selection(fitness_scores))
        # breakpoint()

        c = crossover(p1, p2)
        c = mutate(c, mutation_rate)

        c= get_model_from_chromosome(c, model_class)
        c_fitness = evaluate(c, trainloader)
        heapq.heappop(fitness_scores) #uvijek brisemo najlosijeg
        heapq.heappush(fitness_scores, (c_fitness, gen + random.random(), c))

    #if gen % 10 == 0:
        best_fitness = max(fitness_scores, key=lambda x: x[0])[0]
        best = max(fitness_scores, key=lambda x: x[0])[2]
        a = accuracy(best, testloader) #accuracy racunamo na testnom skupu
        print(f"Generacija {gen + 1}, najbolji fitness: {best_fitness}, najbolji accuracy: {a}")
        gen+=1
    best = max(fitness_scores, key=lambda x: x[0])[2]
    return best  # Vraćamo najbolji model

#generational gp
def genetic_algorithm_generational(population_size, trainloader, testloader, generations=10, elite=0.1, mutation_rate=0.1, model_class=ImageClassifier):
    
    models = initialize_population(population_size, model_class)

    fitness_scores = [(evaluate(model, trainloader), random.random(), model) for model in models]
    fitness_scores.sort(reverse=True, key=lambda x: x[0])
    gen = 0
    elitism_count = int(population_size * elite)
    while gen < generations:
        new_population = []
        elites = [item[2] for item in fitness_scores[:elitism_count]]
        new_population.extend(elites)
        
        while len(new_population) < population_size:
            p1 = get_chromosome_from_model(tournament_selection(fitness_scores))
            p2 = get_chromosome_from_model(tournament_selection(fitness_scores))
            c = crossover(p1, p2)
            c = mutate(c, mutation_rate)
            c = get_model_from_chromosome(c, model_class)
            new_population.append(c)
        
        fitness_scores = [(evaluate(model, trainloader), gen + random.random(), model) for model in new_population]
        fitness_scores.sort(reverse=True, key=lambda x: x[0])
        
        best_fitness = fitness_scores[0][0]
        best_model = fitness_scores[0][2]
        a = accuracy(best_model, testloader)
        print(f"Generacija {gen + 1}, najbolji fitness: {best_fitness}, najbolji accuracy: {a}")
        gen += 1
    
    best_model = fitness_scores[0][2]
    return best_model
#MU+LAMBDA ES