import numpy as np
import random
import os
from fgsm import denorm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sgd import load_model, ImageClassifier


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

def fitness_untargeted(v,model, image, label):
    
    x = v[0]
    y=v[1]
    intensity = v[2]
    
    #image = image.view(image, -1)
    image_denorm = denorm(image)
    #0 - The first image in the batch,0 - The only channel (grayscale),y - Row index (height),x - Column index (width)
    image_perturbed = image_denorm.clone()
    image_perturbed[0, 0, y, x] = intensity #postavljamo pixel na (x,y) poziciji na intensity
    image_perturbed_norm = transforms.Normalize((0.1307,), (0.3081,))(image_perturbed)
    image_perturbed_norm = image_perturbed_norm.view(image_perturbed_norm[0], -1)
    outputs = model(image_perturbed_norm)
    correct_class_prob = outputs[label.item()].item()
    fitness = -correct_class_prob #manja sansa za tocnu klasu = bolji fitness
    
    return fitness
    

def differential_evolution(fitness_strategy, population_size=100, F=0.5, max_gen=100):

    """
        TODO:
        osiguraj da su granice dobre
        implementiraj fitness strategy (targeted attack + untargeted attack)
        napisi return (smisli kako najlakse nac najbolju jedinku u populaciji)
        isprobaj napad
        ...
    """

    model = ImageClassifier()
    load_model(model, "last_trained")

    #inicijaliziraj populaciju
    population = initialize_population(population_size)

    gen = 0
    while gen < max_gen:
        gen+=1
        new_population = []
        for i in range(population_size):
            target = population[i]
            r1, r2, r3 = select(population, i)
            #mutation
            donor = mutation(r1,r2,r3,F)
            #recombination
            trial = recombination(donor, target) 
            #selection
            #u novu populaciju stavi ili trial ili target ovisno koji ima bolji fitness
            if fitness_strategy(trial) > fitness_strategy(target):
                new_population.append(trial)
            else:
                new_population.append(target)
        population = new_population.clone()

    best_fit = None
    best = None
    for v in population:
        fit = fitness_strategy(v)
        if(best_fit == None or fit >= best_fit):
            best_fit = fit
            best = v

    return best, best_fit