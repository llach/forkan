from forkan import ConfigManager
from forkan.rl import load_algorithm

MOC= 'breakout-vae-medium'
RLC = 'breakout-vae-dqn'

cm = ConfigManager([RLC, MOC])

vae = cm.restore_model(MOC, with_dataset=False)

rl_conf = cm.get_config_by_name(RLC)['algorithm']
ev_conf = cm.get_config_by_name(RLC)['environment']

atype = rl_conf.pop('type')
etype = ev_conf.pop('type')

ev_conf['preprocessor'] = vae

alg = load_algorithm(atype, etype, rl_conf, ev_conf)
alg.learn()

