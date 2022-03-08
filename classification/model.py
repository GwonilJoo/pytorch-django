import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # input shape (batch, 3, 26, 26)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # (batch, 16, 26, 26)
            nn.ReLU(), # (batch, 16, 26, 26)
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1), # (batch, 64, 26, 26)
            nn.ReLU(), # (batch, 64, 26, 26)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (batch, 128, 26, 26)
            nn.ReLU(), # (batch, 128, 26, 26)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(128*26*26, 512), # (batch, 512)
            nn.ReLU(),
            nn.Linear(512, 128), # (batch, 128)
            nn.ReLU(),
            nn.Linear(128, 9), # (batch, 9)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(-1, 128*26*26)
        out = self.fc_layer(out)
        return out