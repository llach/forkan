import os
import logging
import tensorflow as tf

from tensorflow.python import debug as tf_debug

from forkan import model_path
from forkan.common.utils import create_dir, rename_latest_run, clean_dir


class BaseAgent(object):

    def __init__(self,
                 env,
                 alg_name,
                 name='default',
                 debug=False,
                 use_tensorboard=True,
                 tensorboard_dir='/tmp/tensorboard/',
                 tensorboard_suffix=None,
                 clean_tensorboard_runs=False,
                 use_checkpoints=True,
                 clean_previous_weights=False,
                 **kwargs,
                 ):
        """
        This class takes care of loading, saving and other initialization stuff that every agent need to do.

        Parameters
        ----------
        env: gym.Environment
            (gym) Environment the agent shall learn from and act on

        alg_name: str
            name of algorithm, e.g. a2c, dqn, trpo etc

        name: str
            descriptive name of this algorithm configuration, e.g. 'atari-breakout'

        debug: bool
            if true, a TensorBoard debugger session is started

        use_tensorboard: bool
            toggles TensorBoard support. If enabled, variable summaries are created and
            written to disk in real time while training.

        tensorboard_dir: str
            Parent directory to save the TensorBoard files to. Within this directory, a new folder is
            created for every training run of the policy. Folders are named as 'run-X' or 'run-latest',
            where X stands for the runs ID.

        tensorboard_suffix: str
            Addition to the foldername for individual runs, could, for example, contain information about
            hyperparamets used. Foldernames will be of the form 'run-SUFFIX-ID'.

        clean_tensorboard_runs: bool
            If true, data of other runs is wiped before execution. This exists mainly to avoid
            disk bloating when testing a lot.

        use_checkpoints: bool
            Saves the model after each episode and upon every policy improvement. A csv-file
            is also written to disk alongside the weights containing information about the run.

        clean_previous_weights: bool
            If true, weights of other runs is wiped before execution. This exists mainly to avoid
            disk bloating when testing a lot.

        """

        # instance name
        self.name = name
        self.alg_name = alg_name

        # environment to act on / learn from
        self.env = env

        # tensorboard and debug related variables
        self.debug = debug
        self.use_tensorboard = use_tensorboard
        self.tensorboard_dir = '{}/{}/{}/'.format(tensorboard_dir, self.alg_name, self.name)
        self.clean_tensorboard_runs = clean_tensorboard_runs
        self.tensorboard_suffix = tensorboard_suffix

        # checkpoint
        self.use_checkpoints = use_checkpoints
        self.clean_previous_weights = clean_previous_weights

        # concat name of instance to path -> distinction between saved instances
        self.checkpoint_dir = '{}/{}/{}/'.format(model_path, self.alg_name, self.name)

        # logger for different levels
        self.logger = logging.getLogger(self.alg_name)

        # global tf.Session and Graph init
        self.sess = tf.Session()

    def _finalize_init(self):
        """ Call this at the end of childs __init__ to setup tensorboard and handle saved checkpoints """

        # init variables
        self.sess.run(tf.global_variables_initializer())

        # launch debug session
        if self.debug:
            self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "localhost:6064")

        # create tensorboard summaries
        if self.use_tensorboard:

            # clean previous runs or add new one
            if not self.clean_tensorboard_runs:
                rename_latest_run(self.tensorboard_dir)
            else:
                clean_dir(self.tensorboard_dir)

            # if there is a directory suffix given, it will be included before the run number in the filename
            tb_dir_suffix = '' if self.tensorboard_suffix is None else '-{}'.format(self.tensorboard_suffix)
            self.tensorboard_dir = '{}/run{}-latest'.format(self.tensorboard_dir, tb_dir_suffix)

            # call child method to do preparations
            self._setup_tensorboard()

            # this operation can be run in a tensorflow session and will return all summaries
            # created above.
            self.merge_op = tf.summary.merge_all()

            self.writer = tf.summary.FileWriter(self.tensorboard_dir,
                                                graph=tf.get_default_graph())

        # flag indicating whether this instance is completely trained
        self.is_trained = False

        # if this instance is working with checkpoints, we'll check whether
        # one is already there. if so, we continue training from that checkpoint,
        # i.e. load the saved weights into target and online network.
        if self.use_checkpoints:

            # remove old weights if needed and not already trained until the end
            if self.clean_previous_weights:
                self.logger.info('Cleaning weights ...')

                if os.path.isfile('{}/done'.format(self.checkpoint_dir)):
                    self.logger.critical('Successfully trained weights shall be deleted under \n\n'
                                         '{}/done\n\n'
                                         'This is most likely a misconfiguration. Either delete the done-file'
                                         ' or the weights themselves manually.'.format(self.checkpoint_dir))

                clean_dir(self.checkpoint_dir)

            # be sure that the directory exits
            create_dir(self.checkpoint_dir)

            # Saver objects handles writing and reading protobuf weight files
            self.saver = tf.train.Saver(var_list=tf.all_variables())

            # file handle for writing episode summaries
            self.csvlog = open('{}/progress.csv'.format(self.checkpoint_dir), 'a')

            # write headline if file is not empty
            if not os.stat('{}/progress.csv'.format(self.checkpoint_dir)).st_size == 0:
                self.csvlog.write('episode, epsilon, reward\n')

            # load already saved weights
            self._load()

    def __del__(self):
        """ Cleanup after object finalization """

        # close tf.Session
        if hasattr(self, 'sess'):
           self.sess.close()

        # close filehandle
        if hasattr(self, 'csvlog'):
            self.csvlog.close()

    def _setup_tensorboard(self):
        raise NotImplementedError

    def _save(self, weight_dir='latest'):
        """ Saves current weights under CHECKPOINT_DIR/weight_dir/ """
        wdir = '{}/{}/'.format(self.checkpoint_dir, weight_dir)
        self.logger.info('Saving weights to {}'.format(wdir))
        self.saver.save(self.sess, wdir)

    def _load(self):
        """
        Loads model weights. If the done-file exists, we know that
        training finished for this set of weights, so we

        """
        # check whether the model being loaded was fully trained
        if os.path.isdir('{}/best/'.format(self.checkpoint_dir)):
            if os.path.isfile('{}/done'.format(self.checkpoint_dir)):
                self.logger.debug('Loading finished weights from {}'.format(self.checkpoint_dir))
            else:
                self.logger.debug('Loading best weights from {}'.format(self.checkpoint_dir))
            self.saver.restore(self.sess, '{}/best/'.format(self.checkpoint_dir))

            # set model as trained
            self.is_trained = True
        elif os.path.isdir('{}/latest/'.format(self.checkpoint_dir)):
            self.logger.warning('Loading pre-trained weights. As this model is not marked as \'done\', \n' +
                                'training will start from t=0 using these weights (this includes filling \n' +
                                'the replay buffer, if one exists). Make sure to have a solved_callback specified to \n' +
                                'avoid training a good policy for too long.')
            self.saver.restore(self.sess, '{}/latest/'.format(self.checkpoint_dir))
        else:
            self.logger.debug('No weights to load found under {}'.format(self.checkpoint_dir))

    def _finalize_training(self):
        """ Takes care of things once training finished """

        # set model as trained
        self.is_trained = True

        # create done-file
        with open('{}/done'.format(self.checkpoint_dir), 'w'): pass

        # close file handle to csv log file
        self.csvlog.close()

    def learn(self):
        raise NotImplementedError

    def run(self, render=True):
        raise NotImplementedError
