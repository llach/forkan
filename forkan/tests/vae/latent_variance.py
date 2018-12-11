import matplotlib.pyplot as plt
import logging
import seaborn as sns; sns.set()
import numpy as np

from forkan.common.config_manager import ConfigManager


"""
Runs model on 1024 different training examples and 
plots the mean variance of the z_i as a bar plot.
"""

logger = logging.getLogger(__name__)

# model name to array: [NUM_PLOTS, NUM_ZI_PER_PLOT]
dataset2plot_shape = {
    'vae-trans': [1, 5],
    'vae-trans-scale': [1, 5],
    'vae-duo': [1, 10],
    'vae-duo-short': [1, 10],
    'breakout': [2, 15],
    'breakout-vae-medium': [2, 15],
}

MODEL_NAME = 'breakout-vae-medium'

# threshold for coloring bars red
COL_THRESH = 0.8

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
    zi[i] = np.exp(model.encode(np.reshape(x_train[i], [1] + list(x_train.shape[1:])))[1])

# average over all three shapes
zi = np.mean(zi, axis=0)

plot_shape = dataset2plot_shape[MODEL_NAME]
num_zi = plot_shape[1]
num_plots = plot_shape[0]

# split zi into multiple bar plots
ys =[]
xs = []
pal = []

for i in range(num_plots):
    xs.append(['z-{}'.format(j+(num_zi*i)) for j in range(num_zi)])
    ys.append(zi[i * num_zi:(i + 1) * num_zi])
    pal.append(['#90D7F3' if k > COL_THRESH else '#F78A8F' for k in ys[-1]])

# create subplots
f, axes = plt.subplots(num_plots, 1, figsize=(9, (6.5*num_plots)))
plt.title('z-i Variances')

# show them heatmaps
if num_plots == 1:
    sns.barplot(x=xs[0], y=ys[0], ax=axes, palette=pal[0], linewidth=0.5)
else:
    for r, ax in enumerate(axes):
        sns.barplot(x=xs[r], y=ys[r], ax=ax, palette=pal[r], linewidth=0.5)

plt.show()

logger.info('Done.')
