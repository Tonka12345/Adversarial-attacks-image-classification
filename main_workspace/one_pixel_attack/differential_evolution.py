import numpy as np
import random
import os
#from fgsm import denorm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sgd import load_model, ImageClassifier

def denorm(batch):
    mean = torch.tensor([0.1307], device=batch.device) #uzimamo 0.1307 za ocekivanje
    std = torch.tensor([0.3081], device=batch.device) #uzimamo 0.3081 za devijaciju
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def initialize_population(population_size):
    #jedinke oblika [x,y,intensity]
    #intensity ~ N(127.5,60)
    #position ~ N(14,7)
    x_positions = np.random.normal(loc=14, scale=7, size=population_size)
    y_positions = np.random.normal(loc=14, scale=7, size=population_size)
    intensities = np.random.normal(loc=127.5, scale=60, size=population_size)
    population = [[x, y, intensity] for x, y, intensity in zip(x_positions, y_positions, intensities)]    
    return population

def select(population, i):
    possible_indexes = [index for index in range(len(population)) if index != i]
    indexes = random.sample(possible_indexes, 3)
    selected = [population[i] for i in indexes]
    return selected

def mutation(r1,r2,r3,F):
    r1 = np.array(r1)
    r2 = np.array(r2)
    r3 = np.array(r3)    
    mutant = r1 + F * (r2 - r3)    
    return mutant.tolist()

def recombination(donor, target, cr = 0.5):
    D = len(donor)
    fromDonor = np.random.randint(0, D) #osigurava da barem jedan index bude is donor vektora
    trial = []
    for j in range(D):
        r = np.random.random()
        if j == fromDonor or r <= cr:
            trial.append(donor[j])
        else:
            trial.append(target[j])
    return trial

def fitness(v,model, image, label, target=None):
    
    x = int(v[0])
    y = int(v[1])
    intensity = v[2]
    
    #ako su kordinate izvan slike vrati los fitness
    height, width = image.shape[2], image.shape[3]
    if x < 0 or x >= width or y < 0 or y >= height:
        return float('-inf')

    #image = image.view(image, -1)
    image_denorm = denorm(image)
    #0 - The first image in the batch,0 - The only channel (grayscale),y - Row index (height),x - Column index (width)
    image_perturbed = image_denorm.clone()
    image_perturbed[0, 0, y, x] = intensity #postavljamo pixel na (x,y) poziciji na intensity
    image_perturbed_norm = transforms.Normalize((0.1307,), (0.3081,))(image_perturbed)
    image_perturbed_norm = image_perturbed_norm.view(image_perturbed_norm.shape[0], -1)
    outputs = model(image_perturbed_norm)
    if target == None:
        correct_class_prob = outputs[0,label].item()
        fitness = -correct_class_prob #manja sansa za tocnu klasu = bolji fitness
    else:
        target_class_prob = outputs[0,target[label]].item()
        fitness = -target_class_prob
    
    return fitness
    

def differential_evolution(model, image, label, population_size=100, F=0.5, max_gen=100, target=None):

    """
        TODO:
        osiguraj da su granice dobre
        implementiraj fitness strategy (targeted attack + untargeted attack)
        ...
    """

    #inicijaliziraj populaciju
    population = initialize_population(population_size)

    gen = 0
    while gen < max_gen:
        gen+=1
        new_population = []
        for i in range(population_size):
            target_v = population[i]
            r1, r2, r3 = select(population, i)
            #mutation
            donor = mutation(r1,r2,r3,F)
            #recombination
            trial = recombination(donor, target_v) 
            #selection
            #u novu populaciju stavi ili trial ili target ovisno koji ima bolji fitness
            if fitness(trial, model, image, label, target) > fitness(target_v,model, image, label, target):
                new_population.append(trial)
            else:
                new_population.append(target_v)
        population = new_population

    best_fit = None
    best = None
    for v in population:
        fit = fitness(v, model, image, label, target)
        if(best_fit == None or fit >= best_fit):
            best_fit = fit
            best = v
    x, y = int(best[0]), int(best[1])
    intensity = best[2]
    image_denorm = denorm(image)
    adv_image = image_denorm.clone()
    if 0 <= x < 28 and 0 <= y < 28:
        adv_image[0, 0, y, x] = intensity
    return adv_image