import matplotlib.pyplot as plt

import numpy as np

import argparse

from bvae import bVAE

from utils import (prune_dataset, load_dsprites, animate_greyscale_dataset,
                   load_dsprites_translational, load_dsprites_duo, load_dsprites_one_fixed)

# load data
x_train, _ = load_dsprites_translational(repetitions=None)

# get image size
image_size = x_train.shape[1]

# load beta-VAE
bvae = bVAE((image_size, image_size, 1), latent_dim=1000, beta=30)
bvae.load('/home/llach/studies/thesis/bvae/trans_b30.0_L1000.h5')

# init array that will hold all heatmaps later
zi = np.empty([x_train.shape[0], 1000], dtype=float)

# forward pass all samples
for i in range(x_train.shape[0]):
    zi[i] = bvae.encode(np.reshape(x_train[i], [1, 64, 64, 1]))[1]

# average over all three shapes
zi = np.mean(zi, axis=0)

zi = np.reshape(zi, [20, 50])

plt.imshow(zi, cmap='Greys_r')

plt.show()

print('Done.')
