import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # input shape (batch, 3, 26, 26)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), # (batch, 64, 26, 26)
            nn.ReLU(), # (batch, 64, 26, 26)
            nn.MaxPool2d(2, 2) # (batch, 64, 13, 13)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (batch, 64, 13, 13) -> 작동 원리 분석할 것
            nn.ReLU(), # (batch, 64, 13, 13)
            nn.UpsamplingNearest2d(scale_factor=2), # (batch, 64, 26, 26)
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, padding=1), # (batch, 3, 26, 26)
            nn.Sigmoid() # (batch, 3, 26, 26)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out