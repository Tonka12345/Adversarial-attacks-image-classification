import torch
import random
import copy
import heapq


def mutate(model, mutation_rate=0.1, mutation_strength=0.1):
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in new_model.parameters(): #mijenjamo tezine i biase
            #print(param.shape)
            if random.random() < mutation_rate:
                param.data += torch.randn_like(param) * mutation_strength
    return new_model

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    with torch.no_grad():
        for p1, p2, c in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
            mask = torch.rand_like(p1) > 0.5  # 50% težina od jednog roditelja, 50% od drugog
            c.copy_(torch.where(mask, p1, p2))
    return child

def evaluate(model, dataloader):
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.NLLLoss()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            #labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()
            #loss = criterion(outputs, labels_one_hot)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return -total_loss  # Manja greška = bolji fitness

def tournament_selection(fitness_scores, k=3):
    tournament = random.sample(fitness_scores, k)  # Nasumično uzimamo k jedinki
    return max(tournament, key=lambda x: x[0])[2] 

def accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def genetic_algorithm(models, dataloader, generations=10, mutation_rate=0.1):
    fitness_scores = [(evaluate(model,dataloader), random.random(), model) for model in models]
    heapq.heapify(fitness_scores)
    gen = 0
    a = 0
    #for gen in range(generations):
    while gen < generations:
        p1 = tournament_selection(fitness_scores)
        p2 = tournament_selection(fitness_scores)
        c = crossover(p1, p2)
        c = mutate(c)

        c_fitness = evaluate(c, dataloader)
        heapq.heappop(fitness_scores)
        heapq.heappush(fitness_scores, (c_fitness, gen + random.random(), c))

        best_fitness = max(fitness_scores, key=lambda x: x[0])[0]
        best = max(fitness_scores, key=lambda x: x[0])[2]
        a = accuracy(best, dataloader)
        print(f"Generacija {gen + 1}, najbolji fitness: {best_fitness}, najbolji accuracy: {a}")
        gen+=1
    best = max(fitness_scores, key=lambda x: x[0])[2]  
      
    return best  # Vraćamo najbolji model
