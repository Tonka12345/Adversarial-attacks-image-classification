import argparse
import torch
from torch.utils.data import DataLoader
from sgd import load_model, testloader
from display_new import visualize_success_rate
from display import visualize_adversarial_examples
from cifar_build_targets import build_target_vector_different_superclass, build_target_vector_same_superclass
from runAttackv2 import runFGSM
from mnist_startegy import MNIST_startegy
from cifar100_strategy import CIFAR100_startegy
import csv
import sys

def save_results_to_csv(epsilons, rates, filepath):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epsilon", "success_rate"])
        for e, r in zip(epsilons, rates):
            writer.writerow([e, r])

def parse_args():
    parser = argparse.ArgumentParser(description="Run FGSM adversarial attack on MNIST or CIFAR-100.")
    parser.add_argument("strategy", choices=["mnist", "cifar100"], help="Dataset/strategy to use")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--targeted", action="store_true", help="Run a targeted attack")
    group.add_argument("--untargeted", action="store_true", help="Run an untargeted attack")

    parser.add_argument("--same", action="store_true", help="(CIFAR100 only) Target vector: same superclass")
    parser.add_argument("--diff", action="store_true", help="(CIFAR100 only) Target vector: different superclass")
    
    parser.add_argument("--graph", action="store_true", help="Run attack for a range of epsilons and plot graph")
    parser.add_argument("--epsilon", type=float, help="Run attack for a single epsilon value")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.strategy == "mnist":
        strategy = MNIST_startegy()
    elif args.strategy == "cifar100":
        strategy = CIFAR100_startegy()
    else:
        raise ValueError("Unsupported strategy.")

    if args.strategy == "mnist" and (args.same or args.diff):
        print("Error: MNIST does not support --same or --diff options.")
        sys.exit(1)

    if args.strategy == "cifar100" and args.targeted and not (args.same or args.diff):
        print("Error: For CIFAR100 targeted attack, specify --same or --diff.")
        sys.exit(1)

    target = None
    if args.targeted:
        if args.strategy == "mnist":
            target = [1,2,3,4,5,6,7,8,9,0]  # Fixed target vector for MNIST
        elif args.strategy == "cifar100":
            if args.same:
                target = build_target_vector_same_superclass()
            elif args.diff:
                target = build_target_vector_different_superclass()

    if args.graph:
        epsilons = [i * 0.01 for i in range(50)]
        rates = []
        for i, epsilon in enumerate(epsilons):
            print(f"Running attack {i+1}/{len(epsilons)} with epsilon={epsilon}")
            successes, total, adv_examples = runFGSM(strategy, epsilon, target=target)
            rate = successes / total
            rates.append(rate)

        save_path = f"./attack_data_NEW/{args.strategy.upper()}/"
        if args.targeted:
            save_path += "targeted/"
        else:
            save_path += "untargeted/"

        visualize_success_rate(epsilons, rates, save_path=save_path)
        save_results_to_csv(epsilons, rates, filepath=save_path + "results.csv")

    elif args.epsilon is not None:
        epsilon = args.epsilon
        successes, total, adv_examples = runFGSM(strategy, epsilon, target=target)
        rate = successes / total
        print(f"epsilon={epsilon}, successful_attacks: {successes} / {total}, success_rate: {rate}")

        if adv_examples:
            print(f"Displaying adversarial examples for epsilon={epsilon}")
            save_path = f"./attack_data_NEW/{args.strategy.upper()}/"
            if args.targeted:
                save_path += "targeted"
            else:
                save_path += "untargeted"
            visualize_adversarial_examples(epsilon, adv_examples, strategy=strategy, save_path=save_path)

    else:
        print("Error: You must specify either --graph or --epsilon.")
        sys.exit(1)

if __name__ == "__main__":
    main()
