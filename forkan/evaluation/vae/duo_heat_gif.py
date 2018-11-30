import matplotlib.pyplot as plt
import logging
import seaborn as sns
import numpy as np
import os

from scipy.ndimage.measurements import label
from forkan.common.config_manager import ConfigManager
from forkan.datasets import load_dataset
from forkan.common.utils import create_dir

"""
For every position of object 1, the activation heatmaps for every position of obejct 2 are computed
and saved as an image. Positions where objects overlap are assigned -1 as activation.
"""

logger = logging.getLogger(__name__)

save_path = os.environ['HOME'] + '/.keras/forkan/figures/duo_heat/'
create_dir(save_path)

cm = ConfigManager()
model = cm.restore_model('vae-duo', with_dataset=False)
dataset = load_dataset('translation')

# we only want the training set
x_train = dataset[0]

# get image size
image_size = x_train.shape[1]

# reshape for better indexing
x_train = np.reshape(x_train, [3, 32, 32, image_size, image_size])

i = 0

for bx in range(0, 32, 2):
    for by in range(0, 32, 2):

        # init array that will hold all heatmaps later
        frames = np.empty([16, 16, model.latent_dim], dtype=float)

        # get base array
        base = x_train[0, bx, by]

        for x in range(0, 32, 2):
            for y in range(0, 32, 2):

                # get array to overlay
                overlay = x_train[2, x, y]

                # new array with 1 wherever one of the base arrays had one
                merged = np.reshape(np.where(overlay != 0, overlay, base), (1, 64, 64, 1))

                # find number of objects in new array
                _, num_objects = label(merged)

                if num_objects > 1:
                    frames[x//2, y//2] = model.encode(merged)[-1]
                else:
                    frames[x//2, y//2] = [-1] * model.latent_dim

        # heatmaps first
        frames = np.rollaxis(frames, -1)

        # create subplots
        f, axes = plt.subplots(2, 5, sharey=True, figsize=(16, 16))

        # show them heatmaps
        for r, ax in enumerate(axes.flat):
            sns.heatmap(frames[r], cbar=False, ax=ax, linewidths=0.0)
            ax.set_aspect('equal', 'box-forced')
            ax.set_title('z{}'.format(r))

        plt.savefig('{}/{}.png'.format(save_path, i))

        i += 1

logger.info('Done.')
