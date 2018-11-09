import matplotlib.pyplot as plt
import logging
import seaborn as sns
import numpy as np

from forkan.config_manager import ConfigManager

logger = logging.getLogger(__name__)

dataset2zi_shape = {
    'bvae-trans': [1, 5],
    'bvae-trans-scale': [1, 5],
    'bvae-duo': [2, 5],
    'bvae-duo-short': [2, 5],
    'bvae-large-latent': [10, 20],
    'bvae-very-large-latent': [20, 50],
}

MODEL_NAME = 'bvae-duo'

cm = ConfigManager()
model, dataset = cm.restore_model(MODEL_NAME)

# we only want the training set
x_train = dataset[0]

if x_train.shape[0] > 1024:
    x_train = x_train[:1024]

# init array that will hold all heatmaps later
zi = np.empty([x_train.shape[0], model.latent_dim], dtype=float)

# forward pass all samples
for i in range(x_train.shape[0]):
    zi[i] = np.exp(model.encode(np.reshape(x_train[i], [1, 64, 64, 1]))[1])

# average over all three shapes
zi = np.mean(zi, axis=0)

zi = np.reshape(zi, dataset2zi_shape[MODEL_NAME])

sns.heatmap(zi, linewidth=0.5)

plt.show()

logger.info('Done.')
