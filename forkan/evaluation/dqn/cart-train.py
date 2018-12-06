from forkan import ConfigManager
from forkan.rl import load_algorithm


def solved_callback(rewards):
    if len(rewards) < 50:
        return False

    for r in rewards[-50:]:
        if r < 195:
            return False

    return True


RLC = 'cart-dqn'

cm = ConfigManager([RLC])

rl_conf = cm.get_config_by_name(RLC)['algorithm']
ev_conf = cm.get_config_by_name(RLC)['environment']

atype = rl_conf.pop('type')
etype = ev_conf.pop('type')

rl_conf['solved_callback'] = solved_callback

alg = load_algorithm(atype, etype, rl_conf, ev_conf)
alg.learn()

