import logging
import yaml
import sys
import os

from forkan import weights_path, config_path
from forkan.rl import load_algorithm, algorithm_list
from forkan.models import model_list, load_model
from forkan.datasets import dataset_list, load_dataset, dataset2input_shape


class ConfigManager(object):

    def __init__(self, config_names=[], file_list=None):
        """
        Loads configurations of models, RL algorithms and pipelines defined in YAML files.

        Parameters
        ----------
        config_names: list(str)
            list of configs (or filenames) to load. if None, all configs are loaded.

        file_list: list(str)
            only search for configs in these files. passing a file here does not result
            in loading all the configs defined in it, but limits the search.
        """

        self.logger = logging.getLogger(__name__)

        self.config_names = config_names
        self.file_list = file_list

        # initially, we have no configs loaded
        self.configs = []

        self.logger.debug('Searching for configuration files in {}'.format(config_path))

        # add extenstions to given file names
        if self.file_list is not None:
            fl_new = []
            for fi in self.file_list:
                if 'yml' not in fi and 'yaml' not in fi:
                    fl_new.append(fi + '.yml')
                else:
                    fl_new.append(fi)
            self.file_list = fl_new

        # search for configs under forkan.config_path
        self.configs = self._load_configs(self.config_names, self.file_list)

        self.logger.debug('Found config files:')
        for cn in self.configs:
            self.logger.debug('     -{}'.format(cn['name']))
        self.logger.debug('Using {} configuration(s).'.format(len(self.configs)))

        # check configs after loading
        self.check()

    def _load_configs(self, load_conf_names, allow_file_names):
        confs = []

        for root, dirs, files in os.walk(config_path):
            for file in files:

                # make sure to have the absolute path to config file
                fi = os.path.abspath(os.path.join(config_path, file))

                # only search in valid yaml files
                if 'yml' not in fi and 'yaml' not in fi:
                    self.logger.debug('Skipping {}.'.format(file))
                    continue

                # config search can be limited to certain files
                # so we skip those who are not given
                if allow_file_names is not None:
                    if not any(file == s for s in allow_file_names):
                        self.logger.debug('{} not in {}. Skipping.'.format(file, allow_file_names))
                        continue

                with open(fi, 'r') as cf:
                    new_configs = [d for d in yaml.load_all(cf)]

                # unify file extensions
                configname = file.replace('.yaml', '').replace('.yml', '')

                # add whole config set
                if configname in load_conf_names or load_conf_names == []:
                    for c in new_configs:
                        confs.append(c)
                    pass

                # select config by name
                for c in new_configs:
                    if c['name'] in load_conf_names:
                        confs.append(c)

        if len(confs) == 0:
            self.logger.error('{} is neither a file nor a config name.'.format(load_conf_names))
            sys.exit(1)

        return confs

    def check(self):
        """
        Checks configs for mandatory fields. If multiple configs are loaded for training,
        this prevents misconfiguration errors in later configs.
        """

        # check for mandatory config parameter
        for conf in self.configs:

            # configs can be either for models or rl algorithms.
            # these two config classes need different checks.
            if 'model' in conf:
                if 'type' not in conf['model']:
                    self.logger.error('No model type given for config {}!'.format(conf['name']))
                    sys.exit(1)
                elif 'type' not in conf['dataset']:
                    self.logger.error('No dataset type given for config {}!'.format(conf['name']))
                    sys.exit(1)
                elif conf['model']['type'] not in model_list:
                    self.logger.error('Unkown model {}'.format(conf['model']['type']))
                    sys.exit(1)
                elif conf['dataset']['type'] not in dataset_list:
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
        """ Execute, i.e. train models defined in configs. """

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
        """ Returns config (dictionary) by name. """

        for conf in self.configs:
            if conf['name'] == config_name:
                return conf

        self.logger.error('Config {} was not found!'.format(config_name))
        sys.exit(1)

    def get_dataset_type(self, config_name):
        """ Returns dataset type of given config. """

        for conf in self.configs:
            if conf['name'] == config_name:
                return conf['dataset']['type']

        self.logger.error('Config {} was not found!'.format(config_name))
        sys.exit(1)

    def restore_model(self, name, with_dataset=True):
        """ Loads model, possibly with dataset, by name. """

        conf = self.get_config_by_name(name)

        if 'model' in conf:

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
