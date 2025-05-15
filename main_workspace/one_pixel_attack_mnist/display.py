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
        
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title(f"Original: {class_names[true_label]}({orig_prob*100:.2f}%)")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(adv_img, cmap='gray')
        axes[1, i].set_title(f"Adversarial: {class_names[adv_pred]}({adv_prob*100:.2f}%)")
        axes[1, i].axis('off')
        
        im = axes[2, i].imshow(perturbation_normalized, cmap='viridis')
        axes[2, i].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    save_filename = os.path.join(save_path, f"one_pixel_attack_new.png")
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    #plt.show()