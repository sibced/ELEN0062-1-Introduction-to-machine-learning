import torch
from torch import nn
from torch.nn.functional import dropout, dropout2d


def conv(in_channels: int, out_channels: int, kernel_size=3) -> nn.Conv2d:
    "returns a convolutional layer that preserves the input shape"
    kern = (kernel_size, kernel_size)
    padd = (kernel_size // 2, kernel_size // 2)
    return nn.Conv2d(in_channels, out_channels, kern, padding=padd)


def dconv(in_channels: int, out_channels: int, kernel_size=3) -> nn.Sequential:
    "retuns a double convoluional layer that halves width and height"
    return nn.Sequential(nn.MaxPool2d((2, 2)),
                         conv(in_channels, out_channels, kernel_size),
                         nn.BatchNorm2d(out_channels), nn.ReLU(True),
                         conv(out_channels, out_channels, kernel_size),
                         nn.BatchNorm2d(out_channels), nn.ReLU(True))


class UConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=True):
        super().__init__()

        layers = [conv(in_channels, out_channels), nn.BatchNorm2d(out_channels),
                  nn.ReLU(True), conv(out_channels, out_channels),
                  nn.BatchNorm2d(out_channels), nn.ReLU(True)]

        if not activation:
            layers = [conv(in_channels, out_channels),
                      nn.BatchNorm2d(out_channels),
                      conv(out_channels, out_channels),
                      nn.BatchNorm2d(out_channels)]

        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(*layers)

    def forward(self, small: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # print(f'in: {small.shape}, {ref.shape}')
        x = torch.cat([ref, self.upsampler(small)], dim=1)
        # print(f'out: {x.shape}')
        return self.conv(x)


class Model(nn.Module):
    """Final architecture for the FCN
    """

    def __init__(self) -> None:
        super().__init__()

        # entry
        self.conv1 = nn.Sequential(conv(3, 32, 3), nn.BatchNorm2d(32),
                                   nn.ReLU(True), conv(32, 32, 3),
                                   nn.BatchNorm2d(32), nn.ReLU(True))

        # downsize: reduce the image size and increase the features
        self.dconv1 = dconv(32, 64, 3)
        self.dconv2 = dconv(64, 128, 3)
        self.dconv3 = dconv(128, 256, 3)
        self.dconv4 = dconv(256, 256, 3)

        # central: deep convolutional layer to 'understand' the feature
        layers = [conv(256, 256, 3),  # nn.BatchNorm2d(256),
                  nn.Dropout(.3), conv(256, 256, 3), nn.BatchNorm2d(256),
                  nn.Dropout(.3), conv(256, 256, 3),  # nn.BatchNorm2d(256),
                  nn.Dropout(.3), conv(256, 256, 3),  # nn.BatchNorm2d(256),
                  nn.Dropout(.3)]
        self.conv_middle = nn.Sequential(*layers)

        # upsize: resize back to original size, and decrease feature
        self.uconv4 = UConv2d(512, 128)
        self.uconv3 = UConv2d(256, 64)
        self.uconv2 = UConv2d(128, 32)
        self.uconv1 = UConv2d(64, 32, activation=False)

        # output: back to one channel and in the (0, 1) domain
        self.conv2 = nn.Sequential(nn.Conv2d(32, 1, (1, 1)), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # first input layer
        x1 = self.conv1(x)

        # go down and keep the indices
        x2 = dropout2d(self.dconv1(x1), .2, self.training)
        x3 = dropout2d(self.dconv2(x2), .5, self.training)
        x4 = dropout2d(self.dconv3(x3), .5, self.training)
        x5 = dropout2d(self.dconv4(x4), .5, self.training)

        # middle layers
        x = self.conv_middle(x5)

        # go up in shape
        x = self.uconv4(x, dropout2d(x4, .5, self.training))
        x = self.uconv3(x, dropout2d(x3, .3, self.training))
        x = self.uconv2(x, dropout2d(x2, .3, self.training))
        x = self.uconv1(x, dropout2d(x1, .3, self.training))

        # output
        return self.conv2(x)
