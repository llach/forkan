from forkan.datasets.mnist import load_mnist
from forkan.datasets.dsprites import load_dsprites, load_dsprites_one_fixed

dataset_list = [
    'mnist',
    'dsprites',
    'translation',
    'translation_scale',
    'dsprites_one_fixed',
    'duo'
]

def load_dataset(dataset_name, kwargs):
    print('Loading dataset {} ...'.format(dataset_name))

    if dataset_name == 'mnist':
        train, val = load_mnist(**kwargs)
        shape = (train[0].shape[1:])
    elif dataset_name == 'dsprites':
        train, val = load_dsprites(type='original', **kwargs)
        shape = (train.shape[1:])
    elif dataset_name == 'translation':
        train, val = load_dsprites(type='translation', **kwargs)
        shape = (train.shape[1:])
    elif dataset_name == 'translation_scale':
        train, val = load_dsprites(type='translation_scale', **kwargs)
        shape = (train.shape[1:])
    elif dataset_name == 'dsprites_one_fixed':
        train, val = load_dsprites_one_fixed(**kwargs)
        shape = (train.shape[1:])
    elif dataset_name == 'duo':
        train, val = load_dsprites(type='duo', **kwargs)
        shape = (train.shape[1:])

    return train, val, shape
