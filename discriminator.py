import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=in_channels*2,
                                              out_channels=features[0],
                                              kernel_size=4,
                                              stride=2,
                                              padding=1,
                                              padding_mode='reflect'),
                                    nn.LeakyReLU(0.2))

        self.block2 = nn.Sequential(nn.Conv2d(in_channels=features[0],
                                              out_channels=features[1],
                                              kernel_size=4,
                                              stride=2,
                                              bias=False,
                                              padding_mode='reflect'),
                                    nn.BatchNorm2d(features[1]),
                                    nn.LeakyReLU(0.2))

        self.block3 = nn.Sequential(nn.Conv2d(in_channels=features[1],
                                              out_channels=features[2],
                                              kernel_size=4,
                                              stride=2,
                                              bias=False,
                                              padding_mode='reflect'),
                                    nn.BatchNorm2d(features[2]),
                                    nn.LeakyReLU(0.2))

        self.block4 = nn.Sequential(nn.Conv2d(in_channels=features[2],
                                              out_channels=features[3],
                                              kernel_size=4,
                                              stride=1,
                                              bias=False,
                                              padding_mode='reflect'),
                                    nn.BatchNorm2d(features[3]),
                                    nn.LeakyReLU(0.2))

        self.block5 = nn.Sequential(nn.Conv2d(in_channels=features[3],
                                              out_channels=1,
                                              kernel_size=4,
                                              stride=1,
                                              padding=1,
                                              padding_mode='reflect'))

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return torch.sigmoid(x)


def main():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)


if __name__ == "__main__":
    main()