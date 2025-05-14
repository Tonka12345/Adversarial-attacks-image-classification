import torch, torchvision

#data_grad je gradijent funkcije gubitka izracunat po ulazu
#radimo fgsm na normaliziranim slikama
def fgsm(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign() # .sign() vraca 1 ako je gradijent pozitivan a -1 ako je negativan
    #dodajemo epsilon u smjeru gradijenta
    # tj. pokusavamo maksimizirati gubitak tako da minimalno(svaki pixel samo za epsilon e [0,1]) promijenimo ulaz
    new_image = image + epsilon * sign_data_grad
    return new_image

def denorm(batch):
    mean = torch.tensor([0.1307], device=batch.device) #uzimamo 0.1307 za ocekivanje
    std = torch.tensor([0.3081], device=batch.device) #uzimamo 0.3081 za devijaciju
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)