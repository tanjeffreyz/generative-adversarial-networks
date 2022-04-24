import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size,
                 activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, out_size),
            activation
        )

    def forward(self, x):
        return self.model.forward(x)


class MnistG(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

        self.model = nn.Sequential(
            LinearBlock(latent_size, 128),
            LinearBlock(128, 256),
            LinearBlock(256, 512),
            LinearBlock(512, 1024),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model.forward(x).view(x.shape[0], 1, 28, 28)


class MnistD(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            LinearBlock(28*28, 512),
            LinearBlock(512, 256),
            LinearBlock(256, 1),
            nn.Sigmoid()            # Sigmoid great for probabilities b/c from 0 to 1
        )

    def forward(self, x):
        return self.model.forward(x)


class Cifar10G(nn.Module):
    def __init__(self, latent_size):
        super().__init__()


class Cifar10D(nn.Module):
    def __init__(self):
        super().__init__()







