import matplotlib.pyplot as plt
import numpy as np

from forkan.config_manager import ConfigManager
from forkan.datasets import load_dataset

cm = ConfigManager('train')
model = cm.restore_model('bvae-duo', with_dataset=False)
dataset = load_dataset('translation')

# we only want the training set
x_train = dataset[0]

# truncate data
x_train = x_train[:1024]

# init array that will hold all heatmaps later
frames = np.empty([32, 32, 10], dtype=float)

set = np.reshape(x_train, [ 32, 32, 64, 64])

# forward pass all samples

for x in range(32):
    for y in range(32):
        frames[x, y] = model.encode(np.reshape(set[x, y], (1, 64, 64, 1)))[-1]

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
