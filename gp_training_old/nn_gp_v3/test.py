import nn
import evolution
import torch

m = nn.ImageClassifier()
ch=evolution.get_chromosome_from_model(m)
m2 = evolution.get_model_from_chromosome(ch, nn.ImageClassifier)
p=list(m.parameters())
p2=list(m2.parameters())

for i in range(len(p)):
    print(torch.all(p[i]==p2[i]))

