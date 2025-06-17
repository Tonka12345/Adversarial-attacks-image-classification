import torch
from torch.utils.data import DataLoader
from sgd import ImageClassifier, load_model, testloader
import torch.nn.functional as F
from display_new import visualize_success_rate
from display import visualize_adversarial_examples#, visualize_success_rate
from cifar_build_targets import build_target_vector_different_superclass, build_target_vector_same_superclass

from runAttackv2 import runFGSM
from mnist_startegy import MNIST_startegy
from cifar100_strategy import CIFAR100_startegy

import sys

import csv

def save_results_to_csv(epsilons, rates, filepath):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epsilon", "success_rate"])
        for e, r in zip(epsilons, rates):
            writer.writerow([e, r])


def main():
    
    strategy1 = MNIST_startegy()
    strategy2 = CIFAR100_startegy()
    epsilons = [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    epsilons3 = [0.35, 0.4, 0.45, 0.5]
    rates = []
    target = [1,2,3,4,5,6,7,8,9,0]
    target_same = build_target_vector_same_superclass()
    target_diff = build_target_vector_different_superclass()
    epsilons4 = [0.03]
    for epsilon in epsilons4:
        successes, total, adv_examples = runFGSM(strategy2, epsilon, target=target_same)
        rate = successes / total
        rates.append(rate)
        print(f"epsilon= {epsilon}, successful_attacks: {successes} / {total}, success_rate: {rate}")
        if adv_examples:
            print(f"Displaying adversarial examples for epsilon={epsilon}")
            visualize_adversarial_examples(epsilon, adv_examples, strategy=strategy2, save_path="./attack_data/CIFAR/targeted")
    #sys.exit(0)

    
    epsilons2 = []
    rates2 = []
    for i in range(50):
        epsilons2.append(i*0.01)
    i = 0
    for epsilon in epsilons2:
        successes, total, adv_examples = runFGSM(strategy1, epsilon, target = target)
        rate = successes / total
        rates2.append(rate)
        print(f"attack {i+1}/50")
        i+=1
    #visiulize_success_rate(epsilons2, rates2, save_path="./attack_data/CIFAR/untargeted")
    visualize_success_rate(epsilons2, rates2, save_path="./attack_data/MNIST/targeted/")
    save_results_to_csv(epsilons2, rates2, "./attack_data/MNIST/targeted/results.csv")

if __name__ == "__main__":
    main()
