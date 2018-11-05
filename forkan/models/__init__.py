import logging

from forkan.models.ae import DenseAE
from forkan.models.bvae import bVAE

logger = logging.getLogger(__name__)

model_list = [
    'ae',
    'bvae'
]


def load_model(model_name, shape, kwargs={}):
    logger.debug('Loading model {} ...'.format(model_name))

    if model_name == 'ae':
        model = DenseAE(shape, **kwargs)
    elif model_name == 'bvae':
        model = bVAE(shape, **kwargs)

    model.compile()

    return model
