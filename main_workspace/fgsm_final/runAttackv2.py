import torch
from torch.utils.data import DataLoader
from sgd import ImageClassifier, load_model, testloader
from fgsm import fgsm, denorm
import torch.nn.functional as F
from torchvision import transforms


def runFGSM(strategy, epsilon, target=None):

    model = strategy.load_model()
    test_loader = strategy.get_test_loader()
    #test_loader mora imati batch_size 1
    #test_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)

    successes = 0
    total = 0
    adv_examples=[]

    for images, labels in test_loader:
        images.requires_grad = True
        input = strategy.prepare_input(images)
        outputs = model(input)
        
        probs, predicted = strategy.get_predictions(outputs)
        #radimo napad samo ako se slika inicijalno dobro klasificira
        if predicted.item() == labels.item():

            total += 1
            loss = strategy.calculate_loss(outputs, labels, target)

            model.zero_grad()#ponisti prosle gradijente            
            loss.backward() #izracunaj nove gradijente
            images_grad = images.grad.data


            adv_images = fgsm(images, epsilon, images_grad)
            adv_input = strategy.prepare_input(adv_images)

            #breakpoint()
            with torch.no_grad():
                new_outputs = model(adv_input)
                new_probs, new_predicted = strategy.get_predictions(new_outputs)
                
            #uspjeh <=> nonteargeted i nije dobra klasifikacija ili targeted i klasifikacija je target 
            if (target == None and new_predicted.item() != labels.item()) or (target!=None and new_predicted.item() == target[labels.item()]):
                #napad je bio uspjesan
                successes += 1
                if len(adv_examples) < 3:
                    images_denorm = strategy.denormalize(images)
                    adv_images_denorm = strategy.denormalize(adv_images)
                    adv_ex = [adv_images_denorm.squeeze().detach().numpy(), new_probs.item()]
                    initial_ex = [images_denorm.squeeze().detach().numpy(), probs.item()]
                    adv_examples.append((adv_ex, initial_ex, new_predicted.item(), labels.item()))
    
    return successes, total, adv_examples