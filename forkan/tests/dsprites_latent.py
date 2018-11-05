import matplotlib.pyplot as plt
import numpy as np

from forkan.config_manager import ConfigManager

cm = ConfigManager('train')
model, dataset = cm.restore_model('bvae-trans')

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
f, axes = plt.subplots(1, 5, sharey=True)

# show them heatmaps
for r in range(5):
    axes[r].imshow(frames[r])
    axes[r].set_title('z{}'.format(r))

plt.show()

print('Done.')