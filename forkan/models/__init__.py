import logging
import sys

from forkan.models.ae.ae import AE
from forkan.models.vae import VAE

logger = logging.getLogger(__name__)

model_list = [
    'ae',
    'vae'
]


def load_model(model_name, shape, kwargs={}):
    logger.debug('Loading model {} ...'.format(model_name))

    if model_name == 'ae':
        model = AE(shape, **kwargs)
    elif model_name == 'vae':
        model = VAE(shape, **kwargs)
    else:
        logger.critical('Model {} not found!'.format(model_name))
        sys.exit(1)

    model.compile()

    return model
