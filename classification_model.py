import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet101


class Classifier(nn.Module):
    def __init__(self, base_model, num_classes=2, **kwargs):
        super(Classifier, self).__init__()
        if base_model == 'resnet18':
            self.base_model = resnet18(**kwargs)
        elif base_model == 'resnet101':
            self.base_model = resnet101(**kwargs)
        else:
            print('Model name {} is not implemented yet!'.format(base_model))
            raise TypeError

        self.fc = nn.Linear(1000, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.base_model(x)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output
