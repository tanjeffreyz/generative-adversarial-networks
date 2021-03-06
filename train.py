import torch
import os
import ssl
import argparse
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from models import MnistG, MnistD, Cifar10D, Cifar10G


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str.lower, choices=('mnist', 'cifar10'))
parser.add_argument(
    '-k', '--k', type=int, default=1,
    help='Trains generator once every K iterations'
)
parser.add_argument('-lr', metavar='lr', type=float, default=1E-4)
args = parser.parse_args()

writer = SummaryWriter()
now = datetime.now()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize based on dataset
if args.dataset == 'mnist':
    dataset = MNIST
    G = MnistG(100)
    D = MnistD()
    epochs = 100
    cp_period = 5
    batch_size = 128
    transform = T.ToTensor()
else:
    dataset = CIFAR10
    G = Cifar10G(100)
    D = Cifar10D()
    epochs = 200
    cp_period = 5
    batch_size = 64
    # transform = T.Compose([
    #     T.ToTensor(),
    #     lambda x: x - torch.mean(x, (1, 2), keepdim=True),
    # ])
    transform = T.ToTensor()
G.to(device)
D.to(device)

ssl._create_default_https_context = ssl._create_unverified_context      # Patch expired certificate error
train_set = dataset(
    root='data', train=True, download=True,
    transform=transform
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
train_iter = iter(train_loader)

# --- Visualize the data ---
# print(next(train_iter)[0][0])
# from matplotlib import pyplot as plt
# from torchvision.utils import make_grid
# grid = make_grid(next(train_iter)[0], nrow=16)
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.imshow(grid.permute(1, 2, 0))
# plt.axis('off')
# plt.tight_layout()
# plt.show()
# exit()

g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

loss_function = torch.nn.BCELoss()

# Create directories
root = os.path.join(
    'models',
    args.dataset,
    now.strftime('%m_%d_%Y'),
    now.strftime('%H_%M_%S')
)
d_weight_dir = os.path.join(root, 'discriminator')
g_weight_dir = os.path.join(root, 'generator')
os.makedirs(d_weight_dir)
os.makedirs(g_weight_dir)
with open(os.path.join(root, 'params.txt'), 'w') as file:
    file.write('\n'.join([
        '\n'.join(f'{k} = {v}' for k, v in vars(args).items()), '\n',
        'Generator:', str(G), '\n',
        'Discriminator:', str(D), ''
    ]))


# Metrics
def save_metrics():
    np.save(os.path.join(root, 'd_losses_fake'), d_losses_fake)
    np.save(os.path.join(root, 'd_losses_real'), d_losses_real)
    np.save(os.path.join(root, 'g_losses'), g_losses)
    np.save(os.path.join(root, 'fake_errors'), fake_errors)
    np.save(os.path.join(root, 'real_errors'), real_errors)


d_losses_fake = np.empty((2, 0))
d_losses_real = np.empty((2, 0))
g_losses = np.empty((2, 0))
fake_errors = np.empty((2, 0))
real_errors = np.empty((2, 0))
for i in tqdm(range(len(train_loader) * epochs), desc='Iteration'):
    ##################################
    #       Train Discriminator      #
    ##################################
    fake_loss = 0
    real_loss = 0
    fake_error = 0
    real_acc = 0
    for _ in range(args.k):
        # Get next batch
        batch = next(train_iter, None)
        if batch is None:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        data = batch[0].to(device)

        real = torch.full((data.shape[0], 1), 1.0).to(device)
        fake = torch.full((data.shape[0], 1), 0.0).to(device)
        noise = torch.Tensor(np.random.normal(size=(data.shape[0], G.latent_size))).to(device)

        d_opt.zero_grad()
        d_real = D.forward(data)
        d_fake = D.forward(G.forward(noise))
        l_fake = loss_function(d_fake, fake)
        l_real = loss_function(d_real, real)
        d_loss = (l_fake + l_real) / 2
        d_loss.backward()
        d_opt.step()

        fake_loss += l_fake.item() / args.k
        real_loss += l_real.item() / args.k
        fake_error += torch.mean(d_fake).item() / args.k
        real_acc += torch.mean(d_real).item() / args.k

        del batch, data, real, fake, noise, d_fake, d_real, l_fake, l_real, d_loss
    d_losses_fake = np.append(d_losses_fake, [[i], [fake_loss]], axis=1)
    d_losses_real = np.append(d_losses_real, [[i], [real_loss]], axis=1)
    fake_errors = np.append(fake_errors, [[i], [fake_error]], axis=1)
    real_errors = np.append(real_errors, [[i], [1 - real_acc]], axis=1)
    writer.add_scalar('Losses/D Fake', fake_loss, i)
    writer.add_scalar('Losses/D Real', real_loss, i)
    writer.add_scalar('Errors/Fake', fake_error, i)
    writer.add_scalar('Errors/Real', 1 - real_acc, i)

    #############################
    #       Train Generator     #
    #############################
    real = torch.full((train_loader.batch_size, 1), 1.0).to(device)
    noise = torch.Tensor(np.random.normal(size=(train_loader.batch_size, G.latent_size))).to(device)

    g_opt.zero_grad()
    g_loss = loss_function(D.forward(G.forward(noise)), real)
    g_loss.backward()
    g_opt.step()

    del real, noise
    g_losses = np.append(g_losses, [[i], [g_loss.item()]], axis=1)
    writer.add_scalar('Losses/Generator', g_loss.item(), i)
    del g_loss

    if i % 3 == 0:
        torch.cuda.empty_cache()

    # Save
    if i % 100 == 0:
        save_metrics()
    if i % (len(train_loader) * cp_period) == 0:
        torch.save(D.state_dict(), os.path.join(d_weight_dir, f'cp_{i}'))
        torch.save(G.state_dict(), os.path.join(g_weight_dir, f'cp_{i}'))

save_metrics()
torch.save(D.state_dict(), os.path.join(d_weight_dir, 'final'))
torch.save(G.state_dict(), os.path.join(g_weight_dir, 'final'))
