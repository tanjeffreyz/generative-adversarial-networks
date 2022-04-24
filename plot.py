import os
import argparse
import numpy as np
from matplotlib import pyplot as plt


TARGET = 'models/mnist/04_23_2022/16_13_56'
STRIDE = 10


d_losses_fake = np.load(os.path.join(TARGET, 'd_losses_fake.npy'))
d_losses_real = np.load(os.path.join(TARGET, 'd_losses_real.npy'))
g_losses = np.load(os.path.join(TARGET, 'g_losses.npy'))
fake_errors = np.load(os.path.join(TARGET, 'fake_errors.npy'))
real_errors = np.load(os.path.join(TARGET, 'real_errors.npy'))

fig, axs = plt.subplots(2, 1)
losses = axs[0]
errors = axs[1]

losses.set(xlabel='Iteration', ylabel='Loss')
losses.plot(d_losses_fake[0][::STRIDE], d_losses_fake[1][::STRIDE], label='D Fake', color='red')
losses.plot(d_losses_real[0][::STRIDE], d_losses_real[1][::STRIDE], label='D Real', color='green')
losses.plot(g_losses[0][::STRIDE], g_losses[1][::STRIDE], label='G', color='orange')
losses.set_ylim(0)
losses.set_xlim(0)
losses.legend()

errors.set(xlabel='Iteration', ylabel='Discriminator Error')
errors.plot(fake_errors[0][::STRIDE], fake_errors[1][::STRIDE], label='Fake', color='red')
errors.plot(real_errors[0][::STRIDE], real_errors[1][::STRIDE], label='Real', color='green')
errors.set_ylim(0, 1)
errors.set_xlim(0)
errors.legend()

plt.tight_layout()
plt.savefig(os.path.join(TARGET, 'metrics'))
plt.show()
