import matplotlib.pyplot as plt

import numpy as np

import argparse

from bvae import bVAE

from utils import (prune_dataset, load_dsprites, animate_greyscale_dataset,
                   load_dsprites_translational, load_dsprites_duo, load_dsprites_one_fixed)

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weights", type=str, default=None)
args = parser.parse_args()

# load data
x_train, _ = load_dsprites_duo()

# get image size
image_size = x_train.shape[1]

# load beta-VAE
bvae = bVAE((image_size, image_size, 1), latent_dim=10, beta=30)
bvae.load('/home/llach/studies/thesis/bvae/duo_beta30.0_L10.h5')

# truncate data
x_train = x_train[:1024]

# init array that will hold all heatmaps later
frames = np.empty([32, 32, 10], dtype=float)

set = np.reshape(x_train, [ 32, 32, 64, 64])

# forward pass all samples

for x in range(32):
    for y in range(32):
        frames[x, y] = bvae.encode(np.reshape(set[x, y], (1, 64, 64, 1)))[-1]



# heatmaps first
frames = np.rollaxis(frames, -1)

# create subplots
f, axes = plt.subplots(1, 10, sharey=True)

# show them heatmaps
for r in range(10):
    axes[r].imshow(frames[r])
    axes[r].set_title('z{}'.format(r))

plt.show()

print('Done.')
