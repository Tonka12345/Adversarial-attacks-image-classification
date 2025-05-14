import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_adversarial_examples(adv_examples, save_path = "./attack_data"):
    
    num_examples = len(adv_examples)
    if num_examples == 0:
        print("No adversarial examples to display.")
        return

    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(3, num_examples, figsize=(4*num_examples, 10))
    
    class_names = [str(i) for i in range(10)]
    
    for i, (adv_img, orig_img, adv_pred, true_label) in enumerate(adv_examples):
        adv_prob = adv_img[1]
        adv_img = adv_img[0]
        orig_prob = orig_img[1]
        orig_img = orig_img[0]

        perturbation = adv_img - orig_img
        perturbation_normalized = (perturbation - np.min(perturbation)) / (np.max(perturbation) - np.min(perturbation))
        epsilon = np.max(np.abs(perturbation))
        
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title(f"Original: {class_names[true_label]}({orig_prob*100:.2f}%)")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(adv_img, cmap='gray')
        axes[1, i].set_title(f"Adversarial: {class_names[adv_pred]}({adv_prob*100:.2f}%)")
        axes[1, i].axis('off')
        
        im = axes[2, i].imshow(perturbation_normalized, cmap='viridis')
        axes[2, i].set_title(f"Perturbation (ε={epsilon:.4f})")
        axes[2, i].axis('off')
    plt.tight_layout()
    plt.suptitle(f"Adversarial Examples ε={epsilon:.4f}", y=0.9, fontsize=16)
    plt.subplots_adjust(top=0.8)

    save_filename = os.path.join(save_path, f"adversarial_examples_epsilon_{epsilon:.4f}.png")
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    #plt.show()


def visiulize_success_rate(epsilons, success_rates, save_path = "./attack_data"):

    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, success_rates, 'bo-', linewidth=2, markersize=8)
    plt.title('FGSM Attack Success Rate vs Epsilon', fontsize=16)
    plt.xlabel('Epsilon', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for x, y in zip(epsilons, success_rates):
        plt.annotate(f'{y:.2f}', 
                    (x, y),
                    textcoords="offset points", 
                    xytext=(0, 5),
                    ha='center')
    
    plt.ylim(0, 1.05)
    
    save_filename = os.path.join(save_path, "attack_success_rate.png")
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    
    #plt.show()
