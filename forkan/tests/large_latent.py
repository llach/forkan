import matplotlib.pyplot as plt
import numpy as np

from forkan.config_manager import ConfigManager

cm = ConfigManager('train')
model, dataset = cm.restore_model('bvae-trans')

# we only want the training set
x_train = dataset[0]

# init array that will hold all heatmaps later
zi = np.empty([x_train.shape[0], 1000], dtype=float)

# forward pass all samples
for i in range(x_train.shape[0]):
    zi[i] = model.encode(np.reshape(x_train[i], [1, 64, 64, 1]))[1]

# average over all three shapes
zi = np.mean(zi, axis=0)

zi = np.reshape(zi, [20, 50])

plt.imshow(zi, cmap='Greys_r')

plt.show()

print('Done.')
