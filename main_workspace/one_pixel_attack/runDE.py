import torch
from torch.utils.data import DataLoader
from sgd import ImageClassifier, load_model, testloader
import torch.nn.functional as F
from torchvision import transforms
#from display import visualize_adversarial_examples, visiulize_success_rate
from differential_evolution import differential_evolution, denorm


def runDE(model, test_loader, target=None):
    #test_loader mora imati batch_size 1
    test_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)

    successes = 0
    total = 0
    adv_examples=[]

    for images, labels in test_loader:
        
        images2 = images.view(images.shape[0], -1)

        outputs = model(images2)
        
        probs, predicted = torch.max(torch.exp(outputs).data, 1)

        #radimo napad samo ako se slika inicijalno dobro klasificira
        if predicted.item() == labels.item():

            total += 1

            images_denorm = denorm(images)

            adv_images = differential_evolution(model,images, labels.item())
            adv_images_norm = transforms.Normalize((0.1307,), (0.3081,))(adv_images) #model prima normalizirane podatke
            adv_images_norm = adv_images_norm.view(adv_images_norm.shape[0], -1)

            new_outputs = model(adv_images_norm)
            new_probs, new_predicted = torch.max(torch.exp(new_outputs).data, 1)
            
            #uspjeh <=> nonteargeted i nije dobra klasifikacija ili targeted i klasifikacija je target 
            if (target == None and new_predicted.item() != labels.item()) or (target!=None and new_predicted.item() == target[labels.item()]):
                #napad je bio uspjesan
                successes += 1
                if len(adv_examples) < 3:
                    adv_ex = [adv_images.squeeze().detach().numpy(), new_probs.item()]
                    initial_ex = [images_denorm.squeeze().detach().numpy(), probs.item()]
                    adv_examples.append((adv_ex, initial_ex, new_predicted.item(), labels.item()))
    
    return successes, total, adv_examples

def main():

    model = ImageClassifier()
    load_model(model, "last_trained")
    
    rates = []
    target = [1,2,3,4,5,6,7,8,9,0]

    #examples

    successes, total, adv_examples = runDE(model, testloader, target=None)
    rate = successes / total
    rates.append(rate)
    print(f"successful_attacks: {successes} / {total}, success_rate: {rate}")
    #if adv_examples:
        #visualize_adversarial_examples(adv_examples, save_path="./attack_data/untargeted")
    
    #rate
    rates2 = []

    successes, total, adv_examples = runDE(model, testloader, target = target)
    rate = successes / total
    rates2.append(rate)
    print(f"success rate: {successes}/{total}, {rate:.4f}")
    #visiulize_success_rate(epsilons2, rates2, save_path="./attack_data/targeted")


if __name__ == "__main__":
    main()
