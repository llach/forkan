import matplotlib.pyplot as plt
import logging
import seaborn as sns
import numpy as np
import sys

from forkan.datasets import load_dataset
from forkan.models import load_model
from forkan import figure_path


"""
Small training script that trains several beta VAEs with different 
beta and |z|. Heatmaps are computed that show variance of the z_i. 
"""
EPOCHS = 150

logger = logging.getLogger(__name__)

# all combinations to test
comb = [(5, 40, [1, 5]),
        (10, 30, [2, 5]),
        (20, 17, [2, 10]),
        (40, 9, [2, 20]),
        (80, 10, [4, 20]),
        (120, 5, [4, 30]),
        (160, 11.5, [4, 40]),
        (200, 10.2, [5, 40]),
        (220, 16.7, [5, 44])]

# load dataset
train, val, input_shape = load_dataset('translation')

# only use a subset for the forward pass
test = train[:500].copy()

# check if heat_shapes match |z|. we don't want the training to fail while we're not watching.
for z_dim, beta, heat_shape in comb:
    if heat_shape[0] * heat_shape[1] is not z_dim:
        logger.critical('{} * {} is not {}!'.format(heat_shape[0], heat_shape[1], z_dim))
        sys.exit(1)

# train each beta - latent dim combination and plot z_i variances
for z_dim, beta, heat_shape in comb:

    # load model
    model = load_model('vae', input_shape, kwargs={'beta': beta, 'latent_dim': z_dim})

    # train it!
    model.fit(train, val, epochs=EPOCHS)

    # init array that will hold all heatmaps later
    zi = np.empty([test.shape[0], model.latent_dim], dtype=float)

    # forward pass all samples
    for i in range(test.shape[0]):
        zi[i] = np.exp(model.encode(np.reshape(test[i], [1, 64, 64, 1]))[1])

    # average over all three shapes
    zi = np.mean(zi, axis=0)

    # shape for plotting
    zi = np.reshape(zi, heat_shape)

    # create & save figure
    sns.heatmap(zi, linewidth=0.5)
    plt.title('(beta) VAE: beta {}; latent {}; epochs {}'.format(beta, z_dim, EPOCHS))
    plt.savefig('{}/zi_L{}_B{}.png'.format(figure_path, z_dim, beta))

    del model

    # this will clear old colorbars
    plt.clf()

