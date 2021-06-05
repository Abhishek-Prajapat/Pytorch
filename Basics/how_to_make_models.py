import torch.nn as nn
from torch.nn import Module
from torchvision import models

# Method 1 : Use nn.Sequential

model_seq = nn.Sequential(nn.Linear(10, 15),
                          nn.ReLU(),
                          nn.Linear(15, 1),
                          nn.Sigmoid())


# Method 2: Use the Module

class NNModel(Module):

    def __init__(self):
        super(NNModel, self).__init__()
        self.l1 = nn.Linear(10, 15)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(15, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.act1(out)
        out = self.l2(out)
        out = self.act2(out)

        return out


model_nn = NNModel()

# Method 3: Use pretrained models

model_pretrained = models.vgg11(pretrained=True)
for layer in model_pretrained.children():
    print(layer)

# print(model_pretrained.avgpool)
# print(model_pretrained.classifier)
'''
As seen in the model.children() the first sequential if for featurization.
To change the model we change the model.avgpool and model.classifier.
'''
