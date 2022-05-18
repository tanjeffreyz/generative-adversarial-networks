import os
import torch
import numpy as np
from torchvision.utils import make_grid
from models import MnistG, Cifar10G
from matplotlib import pyplot as plt


def plot_metrics(target, stride=10):
    d_losses_fake = np.load(os.path.join(target, 'd_losses_fake.npy'))
    d_losses_real = np.load(os.path.join(target, 'd_losses_real.npy'))
    g_losses = np.load(os.path.join(target, 'g_losses.npy'))
    fake_errors = np.load(os.path.join(target, 'fake_errors.npy'))
    real_errors = np.load(os.path.join(target, 'real_errors.npy'))

    fig, axs = plt.subplots(2, 1)
    losses = axs[0]
    errors = axs[1]

    losses.set(xlabel='Iteration', ylabel='Loss')
    losses.plot(d_losses_fake[0][::stride], d_losses_fake[1][::stride], label='D Fake', color='red')
    losses.plot(d_losses_real[0][::stride], d_losses_real[1][::stride], label='D Real', color='green')
    losses.plot(g_losses[0][::stride], g_losses[1][::stride], label='G', color='orange')
    losses.set_ylim(0)
    losses.set_xlim(0)
    losses.legend()

    errors.set(xlabel='Iteration', ylabel='Discriminator Error')
    errors.plot(fake_errors[0][::stride], fake_errors[1][::stride], label='Fake', color='red')
    errors.plot(real_errors[0][::stride], real_errors[1][::stride], label='Real', color='green')
    errors.set_ylim(0, 1)
    errors.set_xlim(0)
    errors.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(target, 'metrics'))
    if SHOW:
        plt.show()


def generate_samples(target, cp):
    if 'mnist' in target:
        model = MnistG(100)
    else:
        model = Cifar10G(100)

    with torch.no_grad():
        state_dict = torch.load(os.path.join(target, 'generator', cp))
        model.load_state_dict(state_dict)
        noise = torch.Tensor(np.random.normal(size=(256, model.latent_size)))
        generated = model.forward(noise)
        grid = make_grid(generated, nrow=16)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()

    samples_dir = os.path.join(target, 'samples')
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    plt.savefig(os.path.join(samples_dir, cp))
    if SHOW:
        plt.show()


if __name__ == '__main__':
    SHOW = False
    TARGET = 'models/cifar10/05_17_2022/18_46_55'
    WEIGHTS = (
        'cp_7820',
        'cp_14076',
        'cp_23460',
        'cp_31280',
        'cp_60996',
        'final'
    )

    plot_metrics(TARGET, stride=100)
    for c in WEIGHTS:
        generate_samples(TARGET, c)
