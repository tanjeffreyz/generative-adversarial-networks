import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size,
                 activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            activation
        )

    def forward(self, x):
        return self.model.forward(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class MnistG(nn.Module):
    def __init__(self, latent_size, linear=False):
        super().__init__()
        self.latent_size = latent_size

        if linear:
            self.model = nn.Sequential(
                nn.Linear(self.latent_size, 128),
                nn.LeakyReLU(negative_slope=0.2),
                LinearBlock(128, 256),
                LinearBlock(256, 512),
                LinearBlock(512, 1024),
                nn.Linear(1024, 28*28),
                nn.Tanh()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(self.latent_size, 128*7*7),
                nn.LeakyReLU(negative_slope=0.2),
                Reshape(128, 7, 7),
                nn.ConvTranspose2d(         # 7x7 -> 14x14
                    128, 128,
                    kernel_size=3, stride=2,
                    padding=1, output_padding=1
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.ConvTranspose2d(         # 14x14 -> 28x28
                    128, 128,
                    kernel_size=3, stride=2,
                    padding=1, output_padding=1
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(128, 1, kernel_size=(3, 3), padding=1),
                nn.Tanh()
            )

    def forward(self, x):
        result = self.model.forward(x)
        return result.view(x.shape[0], 1, 28, 28)


class MnistD(nn.Module):
    def __init__(self, linear=False):
        super().__init__()

        if linear:
            self.model = nn.Sequential(
                nn.Flatten(),
                LinearBlock(28*28, 512),
                LinearBlock(512, 256),
                nn.Linear(256, 1),
                nn.Sigmoid()            # Sigmoid great for probabilities b/c from 0 to 1
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),     # 28x28 -> 14x14
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),     # 14x14 -> 7x7
                nn.LeakyReLU(negative_slope=0.2),
                nn.Flatten(),
                nn.Linear(64*7*7, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.model.forward(x)


class Cifar10G(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size
        self.model = nn.Sequential(
            nn.Linear(self.latent_size, 128*8*8),
            nn.LeakyReLU(negative_slope=0.2),
            Reshape(128, 8, 8),
            nn.ConvTranspose2d(             # 8x8 -> 16x16
                128, 128,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(             # 16x16 -> 32x32
                128, 128,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model.forward(x)


class Cifar10D(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),       # 32x32 -> 16x16
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),      # 16x16 -> 8x8
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
            nn.Linear(64*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model.forward(x)
