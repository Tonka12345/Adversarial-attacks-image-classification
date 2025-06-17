import matplotlib.pyplot as plt
import os
import numpy as np


def visualize_adversarial_examples(epsilon, examples, strategy, save_path="cifar_examples"):

    num_examples = len(examples)
    if num_examples == 0:
        print("No adversarial examples to display.")
        return

    class_names = strategy.get_class_names()

    os.makedirs(save_path, exist_ok=True)

    fig, axs = plt.subplots(3, len(examples), figsize=(3 * len(examples), 9))
    
    if len(examples) == 1:
        axs = axs.reshape(3, 1)

    for i, (adv_ex, orig_ex, adv_label, orig_label) in enumerate(examples):
        adv_img, adv_prob = adv_ex
        orig_img, orig_prob = orig_ex

        perturbation = adv_img - orig_img
        perturbation_normalized = perturbation
        perturbation_normalized = (perturbation - np.min(perturbation)) / (np.max(perturbation) - np.min(perturbation))

        if len(orig_img.shape) == 2:
            orig_img_display = np.stack([orig_img] * 3, axis=-1) 
            adv_img_display = np.stack([adv_img] * 3, axis=-1)
            perturbation_display = np.stack([perturbation_normalized] * 3, axis=-1)
            cmap = 'gray'
            pert_cmap = None 
        else:
            orig_img_display = np.transpose(orig_img, (1, 2, 0))
            adv_img_display = np.transpose(adv_img, (1, 2, 0))
            perturbation_display = np.transpose(perturbation_normalized, (1, 2, 0))
            cmap = None
            pert_cmap = None

        axs[0, i].imshow(orig_img_display, cmap=cmap)
        axs[0, i].set_title(f"Original: {class_names[orig_label]} ({orig_prob:.2f})")
        axs[0, i].axis('off')

        axs[1, i].imshow(adv_img_display, cmap=cmap)
        axs[1, i].set_title(f"Adversarial: {class_names[adv_label]} ({adv_prob:.2f})")
        axs[1, i].axis('off')

        axs[2, i].imshow(perturbation_display, cmap=pert_cmap)
        axs[2, i].set_title(f"Perturbation (ε={epsilon:.4f})")
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.suptitle(f"Adversarial Examples ε={epsilon:.4f}", y=0.96, fontsize=16)
    plt.subplots_adjust(top=0.90)

    save_filename = os.path.join(save_path, f"adversarial_examples_epsilon_{epsilon:.4f}.png")
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')


def visualize_success_rate(epsilons, rates, save_path=None):
    plt.figure(figsize=(8, 6))
    line, = plt.plot(epsilons, rates, marker='o', linewidth=2, markersize=6)
    
    # Add text labels for each data point
    for i, (x, y) in enumerate(zip(epsilons, rates)):
        plt.text(x, y, f"{y:.2f}", 
                ha='center', va='bottom', 
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    plt.title("Success Rate vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Success Rate")
    plt.grid(True)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "success_rate.png"))