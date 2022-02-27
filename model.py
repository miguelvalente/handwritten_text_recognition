import torch
import torch.nn as nn

def conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

class Siamese(nn.Module):
    def __init__(self, n_channels):

        super().__init__()
        self.conv = nn.Sequential(
            conv_block(n_channels, 64, 5, padding='same'),
            conv_block(64, 128, 5, padding='same'),
            conv_block(128, 256, 3, padding='same'),
            nn.Conv2d(256, 512, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(41472, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.fcs = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        return self.fcs(torch.cat((x1, x2), dim=1))

if __name__ == '__main__':
    siam = Siamese(3)
    input = torch.randn(20, 3, 150, 150)
    siam(input, input)