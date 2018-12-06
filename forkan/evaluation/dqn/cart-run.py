from forkan import ConfigManager
from forkan.rl import load_algorithm

RLC = 'cart-dqn'

cm = ConfigManager([RLC])

rl_conf = cm.get_config_by_name(RLC)['algorithm']
ev_conf = cm.get_config_by_name(RLC)['environment']

atype = rl_conf.pop('type')
etype = ev_conf.pop('type')

alg = load_algorithm(atype, etype, rl_conf, ev_conf)
alg.run()

