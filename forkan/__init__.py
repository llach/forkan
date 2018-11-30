import os
import logging
import coloredlogs

from logging.config import dictConfig
from forkan.common.utils import create_dir

weights_path = os.environ['HOME'] + '/.keras/forkan/weights/'
dataset_path = os.environ['HOME'] + '/.keras/datasets/'
config_path = os.path.dirname(os.path.abspath(__file__)) + '/configs/'
figure_path = os.environ['HOME'] + '/.keras/forkan/figures/'
log_file = os.environ['HOME'] + '/.keras/forkan/log.txt'

logging_config = dict(
    version=1,
    formatters={
        'f': {'format':
              '%(asctime)s [%(levelname)-2s] %(name)-4s %(message)s',
              'datefmt': '%H:%M'}
        },
    handlers={
        'h': {'class': 'logging.StreamHandler',
              'formatter': 'f',
              'level': logging.DEBUG}
        },
    root={
        'handlers': ['h'],
        'level': logging.DEBUG,
        },
)

# config for coloredlogs
field_styles = coloredlogs.DEFAULT_FIELD_STYLES
fmt = '%(asctime)s [%(levelname)-8s] %(name)-4s %(message)s'
datefmt = '%H:%M'

# surpress matplotlib debug bloat
logging.getLogger('matplotlib').setLevel(logging.WARNING)

dictConfig(logging_config)
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG', fmt=fmt, datefmt=datefmt)

for d in [weights_path, dataset_path, figure_path]:
    create_dir(d)

# import files and from files that depend on variables defined above after they are defined
from forkan.common.config_manager import ConfigManager
