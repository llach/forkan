import forkan
import yaml
import sys

from forkan.models import model_list, load_model
from forkan.datasets import dataset_list, load_dataset


class ConfigManager(object):

    def __init__(self, active_configs, config_file):

        self.configs = []
        self.active_configs = active_configs
        self.config_file_name = '{}.yml'.format(config_file)
        self.config_file_path = forkan.__file__.replace('__init__.py', '') + self.config_file_name

        with open(self.config_file_path, 'r') as cf:
            self.available_configs = [d for d in yaml.load_all(cf)]

        print('Found {} configs.'.format(len(self.available_configs)))

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
                    print('Could not find config {}'.format(ac))
                    sys.exit(1)
        else:
            self.configs = self.available_configs

        print('Loaded {} configs.'.format(len(self.configs)))

    def check(self):

        # check for mandatory config parameter
        for conf in self.configs:
            if 'name' not in conf['model']:
                print('No model name given for config {}!'.format(conf['name']))
                sys.exit(1)
            elif 'name' not in conf['dataset']:
                print('No dataset name given for config {}!'.format(conf['name']))
                sys.exit(1)
            elif conf['model']['name'] not in model_list:
                print('Unkown model {}'.format(conf['model']['name']))
                sys.exit(1)
            elif conf['dataset']['name'] not in dataset_list:
                print('Unkown dataset {}'.format(conf['dataset']['name']))
                sys.exit(1)

        print('Checks were successful.')


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


