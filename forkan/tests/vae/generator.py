import matplotlib.pyplot as plt
import numpy as np
import os

from forkan.common.config_manager import ConfigManager
from forkan.datasets import dataset2input_shape
from forkan.common.utils import create_dir


"""
Decoder of AE like model is run where all z_i but one are kept fixed. The variable z_i takes on
values from a certain range and output images are generated.
"""

MODEL_NAME = 'breakout-vae-medium'

cm = ConfigManager()
model = cm.restore_model(MODEL_NAME, with_dataset=False)

idx = 28

saving = True
save_path = os.environ['HOME'] + '/.keras/forkan/figures/{}/z_{}/'.format(MODEL_NAME, idx)
create_dir(save_path)

np.random.seed(1)

latents = np.random.normal(0, 1, model.latent_dim)

for i, r in enumerate(np.linspace(-3, 3, 16)):
    latents[idx] = r

    img = model.decode(np.reshape(latents, [1, model.latent_dim]))
    target_shape = dataset2input_shape[cm.get_dataset_type(MODEL_NAME)]

    if target_shape[-1] == 1:
        target_shape = target_shape[:-1]

    plt.imshow(np.reshape(img, target_shape), cmap='Greys_r')
    plt.title('z_{}'.format(idx))

    if saving:
        plt.savefig('{}/{}.png'.format(save_path, i))

    plt.pause(0.1)

    plt.clf()
    i += 1

plt.show()
