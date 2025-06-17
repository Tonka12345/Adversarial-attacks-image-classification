import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from sgd import ImageClassifier, load_model, testloader
from torch.utils.data import DataLoader


class MNIST_startegy:
    def __init__(self, model_path = "last_trained"):
        self.mean = 0.1307
        self.std = 0.3081
        self.model_path = model_path
    
    def get_test_loader(self):
        test_loader = DataLoader(testloader.dataset, batch_size=1, shuffle=False)
        return test_loader
    
    def denormalize(self, batch):
        mean = torch.tensor([self.mean], device=batch.device) #uzimamo 0.1307 za ocekivanje
        std = torch.tensor([self.std], device=batch.device) #uzimamo 0.3081 za devijaciju
        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    
    def normalize(self, batch):
        return transforms.Normalize((self.mean,), (self.std,))(batch)
    
    def get_num_classes(self):
        return 10
    
    def prepare_input(self, images):
        return images.view(images.shape[0], -1)
    
    def load_model(self):
        model = ImageClassifier()
        load_model(model, self.model_path)
        model.eval()
        return model
    
    def get_predictions(self, outputs):
        probs, predicted = torch.max(torch.exp(outputs).data, 1)
        return probs, predicted
    
    def calculate_loss(self, outputs, labels, target=None):
        if target is None:
            loss = F.nll_loss(outputs, labels)
        else:
            loss = outputs[0, target[labels.item()]]
        return loss
    
    def get_class_names(self):
        return [0,1,2,3,4,5,6,7,8,9]
