class LinearSchedule(object):
    def __init__(self, max_t, final, initial=1.0):
        """
        Anneals from inital value to final value in max_t timesteps.
        Copied from OpenAI baselines to avoid big dependency.
        """
        self.max_t = max_t
        self.final = final
        self.initial = initial

    def value(self, t):
        frac = min(float(t) / self.max_t, 1.0)
        return self.initial + frac * (self.final - self.initial)
