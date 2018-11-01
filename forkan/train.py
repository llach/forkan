import argparse
import sys

from forkan.config_manager import ConfigManager

parser = argparse.ArgumentParser()
parser.add_argument('configs', nargs='*', type=str, default=None)
args = parser.parse_args()

cm = ConfigManager(args.configs, 'train')
cm.check()
cm.exec()
