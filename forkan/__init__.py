import os
import errno

weights_path = os.environ['HOME'] + '/.keras/forkan/weights/'
dataset_path = os.environ['HOME'] + '/.keras/datasets/'

for dir in [weights_path, dataset_path]:
    if not os.path.isdir(dir):
        try:
            os.makedirs(dir)
            print('Creating {}'.format(dir))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(dir):
                pass
