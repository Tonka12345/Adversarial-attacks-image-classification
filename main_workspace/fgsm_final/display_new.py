import matplotlib.pyplot as plt
import os
import numpy as np
import torch


def visualize_success_rate(epsilons, rates, save_path=None):
    plt.figure(figsize=(8, 6))
    line, = plt.plot(epsilons, rates, marker='o', linewidth=2, markersize=6)
    
    # # Add text labels for each data point
    # for i, (x, y) in enumerate(zip(epsilons, rates)):
    #     plt.text(x, y, f"{y:.2f}", 
    #             ha='center', va='bottom', 
    #             fontsize=9,
    #             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    plt.title("Success Rate vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Success Rate")
    plt.grid(True)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "success_rate.png"))