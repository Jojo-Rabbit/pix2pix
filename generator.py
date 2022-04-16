import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, down=True, act='relu', drop=False):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                             out_channels=out_channel,
                                             kernel_size=4,
                                             stride=2,
                                             padding=1,
                                             bias=False,
                                             padding_mode='reflect') if down
                                   else nn.ConvTranspose2d(in_channels=in_channel,
                                                           out_channels=out_channel,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU() if act == 'relu'
                                   else nn.LeakyReLU(0.2))
        self.drop = drop
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.drop else x


class Generator(nn.Module):
    def __init__(self, in_channel=3, feature=64):
        super().__init__()
        self.initial = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                               out_channels=feature,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1,
                                               padding_mode='reflect'),
                                     nn.LeakyReLU(0.2))

        self.down1 = Block(feature, feature * 2, True, 'Leaky', False)
        self.down2 = Block(feature * 2, feature * 4, True, 'Leaky', False)
        self.down3 = Block(feature * 4, feature * 8, True, 'Leaky', False)
        self.down4 = Block(feature * 8, feature * 8, True, 'Leaky', False)
        self.down5 = Block(feature * 8, feature * 8, True, 'Leaky', False)
        self.down6 = Block(feature * 8, feature * 8, True, 'Leaky', False)

        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=feature * 8,
                                                  out_channels=feature * 8,
                                                  kernel_size=4,
                                                  stride=2,
                                                  padding=1,
                                                  padding_mode='reflect'),
                                        nn.ReLU())

        self.up1 = Block(feature * 8, feature * 8, False, 'relu', True)
        self.up2 = Block(feature * 8 * 2, feature * 8, False, 'relu', True)
        self.up3 = Block(feature * 8 * 2, feature * 8, False, 'relu', True)
        self.up4 = Block(feature * 8 * 2, feature * 8, False, 'relu', False)
        self.up5 = Block(feature * 8 * 2, feature * 4, False, 'relu', False)
        self.up6 = Block(feature * 4 * 2, feature * 2, False, 'relu', False)
        self.up7 = Block(feature * 2 * 2, feature, False, 'relu', False)
        self.final = nn.Sequential(nn.ConvTranspose2d(in_channels=feature * 2,
                                                      out_channels=in_channel,
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding=1),
                                   nn.Tanh())

    def forward(self, x):
        d1 = self.initial(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        final = self.final(torch.cat([up7, d1], 1))

        return final


def main():
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    output = model(x)
    print(output.shape)


if __name__ == "__main__":
    main()
