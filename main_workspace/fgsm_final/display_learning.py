import matplotlib.pyplot as plt
import numpy as np
import os

def display_learning(epochs, accuracies_test, accuracies_train, losses_test, losses_train, save_path='./learning_data'):
    
    x_values = list(range(1, epochs + 1))

    os.makedirs(save_path, exist_ok=True)

    #losses
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, losses_train, 'b-', linewidth=2, marker='o', markersize=6, label='Training Loss')
    plt.plot(x_values, losses_test, 'r-', linewidth=2, marker='s', markersize=6, label='Test Loss')
    
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Test Loss during Epochs', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.xticks(x_values)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss.png'), dpi=300)
    
    #accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, accuracies_train, 'g-', linewidth=2, marker='o', markersize=6, label='Training Accuracy')
    plt.plot(x_values, accuracies_test, 'm-', linewidth=2, marker='s', markersize=6, label='Test Accuracy')
    
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Training and Test Accuracy during Epochs', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.xticks(x_values)
    
    if max(max(accuracies_train), max(accuracies_test)) <= 1.0:
        plt.ylim(0, 1.0)
        plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'accuracy.png'), dpi=300)