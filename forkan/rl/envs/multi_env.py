import numpy as np
import os
import logging

from forkan import fixed_seed
from forkan.rl import EnvWrapper
from multiprocessing import Process, Pipe


def worker(parent, conn, tid, env):
    """ Waits until it receives a command via Pipe. """

    # we don't need this end of the pipe
    parent.close()

    # save PID and change seed based on this
    pid = os.getpid()

    # thread ID as seed: runs will yield same result for same number of threads
    # while maintaining different seeds across threads
    # process ID as seed: IDs change from run to run
    if fixed_seed:
        np.random.seed(tid)
    else:
        np.random.seed(pid)

    try:
        while True:
            re = conn.recv()
            cmd, data = re
            if cmd == 'print_seed':
                s = np.random.get_state()[1][0]
                print('Thread {} has seed {}'.format(tid, s))
            elif cmd == 'step':
                ob, reward, done, info = env.step(data)

                # as this worker is supposed for threaded multistep environments,
                # we reset immediatly after a done. Be sure to only use discounting
                # that takes dones into consideration.
                if done:
                    ob = env.reset()
                conn.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                conn.send(ob)
            elif cmd == 'render':
                conn.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                conn.close()
                break
            elif cmd == 'get_spaces':
                conn.send((env.observation_space, env.action_space))
            elif cmd == 'pid':
                print(pid)
            else:
                print('Thread {} received unknown command \'{}\''.format(tid, cmd))
    except KeyboardInterrupt:
        print('Thread {} got KeyboardInterrupt'.format(tid))
    finally:
        conn.close()


class MultiEnv(EnvWrapper):

    def __init__(self, num_envs, make_env):
        """
        Multithreads environments.

        Parameters
        ----------
        num_envs: int
            number of threads to run environments in

        make_env: function
            functions that returns configured env

        """

        # inheriting from EnvWrapper and passing it an env makes spaces available.
        super().__init__(make_env())

        self.ps = []
        self.conns = []

        self.num_envs = num_envs

        # create process managing environments
        for i in range(num_envs):

            # create env
            e = make_env()

            # create process and communication pipe
            parent_conn, child_conn = Pipe()
            p = Process(target=worker, args=(parent_conn, child_conn, i, e,))

            p.daemon = True
            p.start()

            child_conn.close()

            self.ps.append(p)
            self.conns.append(parent_conn)

        self.logger.info('Spawned {} environments.'.format(num_envs))

    def __del__(self):
        """ Close all pipes, join all processes. """

        for con in self.conns:
            con.close()

        for p in self.ps:
            p.join()

    def cmd(self, cmd):
        """ Executes command in all envs. """

        for con in self.conns:
            con.send((cmd, None))

    def reset(self):
        """ Resets all environments, returning the observation matrix. """

        # reset envs
        for p in self.conns:
            p.send(('reset', None))

        # collect obs
        obs = [p.recv() for p in self.conns]

        return np.stack(obs)

    def step(self, actions):
        assert len(actions) == len(self.ps), 'Give {} actions'.format(len(self.ps))

        # send commands to processes
        for con, act in zip(self.conns, actions):
            con.send(('step', act))

        # collect results
        results = [p.recv() for p in self.conns]
        obs, rs, ds, ins = zip(*results)

        return np.stack(obs), np.stack(rs), np.stack(ds), ins
