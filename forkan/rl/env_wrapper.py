class EnvWrapper(object):

    def __init__(self, env):
        self.env = env

    def __getattr__(self, item):
        return getattr(self.env, item)
