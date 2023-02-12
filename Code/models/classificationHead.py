import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, num_features=None, num_classes=None):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.fc = nn.Linear(self.num_features, self.num_classes)


    def forward(self, x):
        return self.fc(x)