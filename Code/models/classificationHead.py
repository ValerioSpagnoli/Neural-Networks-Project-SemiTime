import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, num_features=None, num_classes=None):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.fc = nn.Linear(self.num_features, self.num_classes)
        #self.fc1 = nn.Linear(self.num_features, 256)
        #self.bn1 = nn.BatchNorm1d(256)
        #self.leakyReLU = nn.LeakyReLU()
        #self.fc2 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.fc(x)
        #x = self.fc1(x)
        #x = self.bn1(x)
        #x = self.leakyReLU(x)
        #x = self.fc2(x)
        return x