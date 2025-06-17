import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models



class CIFAR100_startegy:
    def __init__(self, model_path = None):
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
    
    def get_test_loader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        testset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
        
        return testloader
    
    def denormalize(self, batch):
        mean = torch.tensor(self.mean, device=batch.device)
        std = torch.tensor(self.std, device=batch.device)
        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    
    def normalize(self, batch):
        return transforms.Normalize(self.mean, self.std)(batch)
    
    def get_num_classes(self):
        return 100
    
    def prepare_input(self, images):
        return images #ne treba flattening kao kod mnista
    
    def load_model(self, model_path="cifar100_model.pth"):
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet20', pretrained=False)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_predictions(self, outputs):
        probs, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
        return probs, predicted
    
    def calculate_loss(self, outputs, labels, target=None):
        if target is None:
            loss = F.cross_entropy(outputs, labels)
        else:
            target_tensor = torch.tensor([target[labels.item()]])
            loss = -F.cross_entropy(outputs, target_tensor)
        return loss
    
    def get_class_names(self):
        testset = torchvision.datasets.CIFAR100(root='./', train=False, download=True)
        return testset.classes