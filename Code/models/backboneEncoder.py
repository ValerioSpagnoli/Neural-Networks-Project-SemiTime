import torch.nn as nn

class BackboneEncoder(nn.Module):
    def __init__(self, num_features=None):
        super().__init__()
        self.num_features = num_features

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.num_features),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)
        )

        self.flatten = nn.Flatten()

        
    def forward(self, x):
        
        x = x.view(x.shape[0], 1, -1) # shape = (batch_size, 1, num_col)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = nn.functional.normalize(x, dim=1)
        
        return x
