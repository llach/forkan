import logging
import yaml
import sys
import os

from forkan import weights_path, config_path
from forkan.rl import load_algorithm, algorithm_list
from forkan.models import model_list, load_model
from forkan.datasets import dataset_list, load_dataset, dataset2input_shape


class ConfigManager(object):

    def __init__(self, config_names=[]):

        self.logger = logging.getLogger(__name__)

        self.config_names = config_names
        self.configs = []

        self.logger.debug('Searching for configuration files in {}'.format(config_path))

        self.logger.debug('Found configs:')
        for root, dirs, files in os.walk(config_path):
            for file in files:
                fi = os.path.abspath(os.path.join(config_path, file))

                if 'yml' not in fi and 'yaml' not in fi:
                    self.logger.debug('Skipping {}.'.format(file))
                    continue

                self.logger.debug('    - {}'.format(file))

                with open(fi, 'r') as cf:
                    new_configs = [d for d in yaml.load_all(cf)]

                configname = file.replace('.yaml', '').replace('.yml', '')

                # add whole config set
                if configname in self.config_names or self.config_names == []:
                    for c in new_configs:
                        self.configs.append(c)
                    pass

                # select config by name
                for c in new_configs:
                    if c['name'] in self.config_names:
                        self.configs.append(c)

        if len(self.configs) == 0:
            self.logger.error('{} is neither a file nor a config name.'.format(config_names))
            sys.exit(1)

        self.logger.debug('Using {} configuration(s).'.format(len(self.configs)))

        # check configs after loading
        self.check()

    def check(self):

        # check for mandatory config parameter
        for conf in self.configs:

            # configs can be either for models or rl algorithms.
            # these two config classes need different checks.
            if 'model' in conf:
                if 'type' not in conf['model']:
                    self.logger.error('No model type given for config {}!'.format(conf['name']))
                    sys.exit(1)
                elif 'name' not in conf['dataset']:
                    self.logger.error('No dataset name given for config {}!'.format(conf['name']))
                    sys.exit(1)
                elif conf['model']['type'] not in model_list:
                    self.logger.error('Unkown model {}'.format(conf['model']['type']))
                    sys.exit(1)
                elif conf['dataset']['name'] not in dataset_list:
                    self.logger.error('Unkown dataset {}'.format(conf['dataset']['type']))
                    sys.exit(1)

            elif 'algorithm' in conf:
                if 'type' not in conf['algorithm']:
                    self.logger.critical('No algorithm type provided for {}!'.format(conf['name']))
                    sys.exit(1)
                elif 'type' not in conf['environment']:
                    self.logger.critical('No environment id provided for {}!'.format(conf['name']))
                    sys.exit(1)
                elif conf['algorithm']['type'] not in algorithm_list:
                    self.logger.error('Unkown algorithm {}'.format(conf['algorithm']['type']))
                    sys.exit(1)

            else:
                self.logger.critical('Invalid configuration wo?{}!'.format(conf['name']))

        self.logger.debug('Checks were successful.')

    def exec(self):

        # execute configured actions
        for conf in self.configs:

            if 'model' in conf:

                # load dataset along with input shape
                dataset_type = conf['dataset'].pop('type')
                train, val, input_shape = load_dataset(dataset_type, conf['dataset'])


                # load model with dataset specific input shape
                model_type = conf['model'].pop('type')
                model = load_model(model_type, input_shape, conf['model'])

                # train model
                model.fit(train, val, **conf['training'])

                # save weights
                model.save(dataset_type)

                # restore popped values for later use
                conf['dataset']['type'] = dataset_type
                conf['model']['type'] = model_type

            elif 'algorithm' in conf:

                alg_type = conf['algorithm'].pop('type')
                env_type = conf['environment'].pop('type')

                # load configured algorithm & environment
                alg = load_algorithm(alg_type, env_type, conf['algorithm'], conf['environment'])

                # learn agent on environment
                alg.learn()

                # restore popped values for alter use
                conf['algorithm']['type'] = alg_type
                conf['environment']['type'] = env_type

    def get_config_by_name(self, config_name):

        for conf in self.configs:
            if conf['name'] == config_name:
                return conf

        self.logger.error('Config {} was not found!'.format(config_name))
        sys.exit(1)

    def get_dataset_type(self, config_name):

        for conf in self.configs:
            if conf['name'] == config_name:
                return conf['dataset']['type']

        self.logger.error('Config {} was not found!'.format(config_name))
        sys.exit(1)

    def restore_model(self, name, with_dataset=True):

        conf = self.get_config_by_name(name)

        # get model and dataset name
        dataset_type = conf['dataset'].pop('type')

        if with_dataset:
            train, val, input_shape = load_dataset(dataset_type, conf['dataset'])
        else:
            input_shape = dataset2input_shape[dataset_type]

        model_type = conf['model'].pop('type')

        # restore weight path from config
        weights = '{}/{}_{}_b{}_L{}_E{}.h5'.format(weights_path, model_type, dataset_type,conf['model']['beta'],
                                                   conf['model']['latent_dim'], conf['training']['epochs'])

        model = load_model(model_type, input_shape, conf['model'])
        model.load(weights)

        # restore popped values for later use
        conf['dataset']['type'] = dataset_type
        conf['model']['type'] = model_type

        if with_dataset:
            return model, (train, val)
        else:
            return model