import matplotlib.pyplot as plt
import logging
import seaborn as sns
import numpy as np

from forkan.common.config_manager import ConfigManager

""" 
For models trained on the dsprites dataset with only translation as generative factors.
Will produce a heatmap of activations for each possible position of an object and for
each z_i. The mean of three activations for each position is calculated.
"""

logger = logging.getLogger(__name__)

cm = ConfigManager()
model, dataset = cm.restore_model('vae-trans')

# init array that will hold all heatmaps later
frames = np.empty([3, 32, 32, 5], dtype=float)

# reshape the set for nice iteration
dataset = np.reshape(dataset[0], [3, 32, 32, 64, 64])

# forward pass all samples
for i in range(3):
    for x in range(32):
        for y in range(32):
            frames[i, x, y] = model.encode(np.reshape(dataset[i, x, y], (1, 64, 64, 1)))[-1]

# average over all three shapes
frames = np.mean(frames, axis=0)

# heatmaps first
frames = np.rollaxis(frames, -1)

# create subplots
f, axes = plt.subplots(1, 5, sharey=True, figsize=(16, 8))

# show them heatmaps
for r, ax in enumerate(axes.flat):
    sns.heatmap(frames[r], cbar=False, ax=ax)
    ax.set_aspect('equal', 'box-forced')
    ax.set_title('z{}'.format(r))

plt.show()

logger.info('Done.')
