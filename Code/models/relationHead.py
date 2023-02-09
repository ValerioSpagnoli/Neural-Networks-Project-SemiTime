import torch.nn as nn

class RelationHead(nn.Module):
    def __init__(self, num_features=None):
        super().__init__()
        self.num_features = num_features
        self.fc1 = nn.Linear(2*self.num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.leakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leakyReLU(x)
        x = self.fc2(x)
        
        return x