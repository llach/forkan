import matplotlib.pyplot as plt
import numpy as np

from forkan.config_manager import ConfigManager
from forkan.datasets import dataset2input_shape

MODEL_NAME = 'breakout'

cm = ConfigManager()
model = cm.restore_model(MODEL_NAME, with_dataset=False)

np.random.seed(1)

idx = 13
latents = np.random.normal(0, 1, (model.latent_dim))

for r in np.linspace(-3, 3, 16):
    latents[idx] = r

    img = model.decode(np.reshape(latents, [1, model.latent_dim]))
    plt.imshow(np.reshape(img, dataset2input_shape[cm.get_dataset_name(MODEL_NAME)]), cmap='Greys_r')
    plt.pause(0.1)

plt.show()
