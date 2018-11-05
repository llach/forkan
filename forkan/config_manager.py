import logging
import forkan
import yaml
import sys

from forkan import weights_path
from forkan.models import model_list, load_model
from forkan.datasets import dataset_list, load_dataset, dataset2input_shape


class ConfigManager(object):

    def __init__(self, config_file, active_configs=[]):

        self.logger = logging.getLogger(__name__)

        self.configs = []
        self.active_configs = active_configs
        self.config_file_name = '{}.yml'.format(config_file)
        self.config_file_path = forkan.__file__.replace('__init__.py', '') + self.config_file_name

        with open(self.config_file_path, 'r') as cf:
            self.available_configs = [d for d in yaml.load_all(cf)]

        self.logger.debug('Found {} configs.'.format(len(self.available_configs)))

        # save only configs we want to use; if none, we use all
        if len(active_configs) > 0:
            for ac in self.active_configs:

                found = False

                for conf in self.available_configs:
                    if conf['name'] == ac:
                        self.configs.append(conf)
                        found = True
                        break

                if not found:
                    self.logger.error('Could not find config {}'.format(ac))
                    sys.exit(1)
        else:
            self.configs = self.available_configs

        self.logger.debug('Loaded {} configs.'.format(len(self.configs)))

        # check configs after loading
        self.check()

    def check(self):

        # check for mandatory config parameter
        for conf in self.configs:
            if 'name' not in conf['model']:
                self.logger.error('No model name given for config {}!'.format(conf['name']))
                sys.exit(1)
            elif 'name' not in conf['dataset']:
                self.logger.error('No dataset name given for config {}!'.format(conf['name']))
                sys.exit(1)
            elif conf['model']['name'] not in model_list:
                self.logger.error('Unkown model {}'.format(conf['model']['name']))
                sys.exit(1)
            elif conf['dataset']['name'] not in dataset_list:
                self.logger.error('Unkown dataset {}'.format(conf['dataset']['name']))
                sys.exit(1)

        self.logger.debug('Checks were successful.')

    def exec(self):

        # execute configured actions
        for conf in self.configs:

            # load dataset along with input shape
            dataset_name = conf['dataset'].pop('name')
            train, val, input_shape = load_dataset(dataset_name, conf['dataset'])


            # load model with dataset specific input shape
            model_name = conf['model'].pop('name')
            model = load_model(model_name, input_shape, conf['model'])

            # train model
            model.fit(train, val, **conf['training'])

            # save weights
            model.save(dataset_name)

    def get_config_by_name(self, name):

        for conf in self.configs:
            if conf['name'] == name:
                return conf

        self.logger.error('Config {} was not found!'.format(name))
        sys.exit(1)

    def restore_model(self, name, with_dataset=True):

        conf = self.get_config_by_name(name)

        # get model and dataset name
        dataset_name = conf['dataset'].pop('name')

        if with_dataset:
            train, val, input_shape = load_dataset(dataset_name, conf['dataset'])
        else:
            input_shape = dataset2input_shape[dataset_name]

        model_name = conf['model'].pop('name')

        # restore weight path from config
        weights = '{}/{}_{}_b{}_L{}_E{}.h5'.format(weights_path, model_name, dataset_name,conf['model']['beta'],
                                                   conf['model']['latent_dim'], conf['training']['epochs'])

        model = load_model(model_name, input_shape, conf['model'])
        model.load(weights)

        if with_dataset:
            return model, (train, val)
        else:
            return model
