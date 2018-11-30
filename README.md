# forkan

This repository contains implementations of selected deep learning models and reinforcement algorithms.

Additional README files can be found in some model and algorithm directories that go more into detail
on the implementation of this particular method.

## Package Structure

`common` provides functions that are used in multiple subpackages

`configs` contains model and policy configuration files in yaml format

`datasets` defines loading, generation and other tools regarding datasets

`evaluation` scripts evaluating models and policies 

`models` implementation of different (deep) models

`rl` contains reinforcement learning algorithms

## Installation

It is recommended to use a python virtual environment. This way, faulty installations and enivronment misconfigurations
can be reverted quite easily. To create a virtual env, simply run
```bash
python3.5 -m venv ENV_NAME
```
where `ENV_NAME` is replaced by the name of the virtual env.
Using the venv is as simple as running
```bash
source ENV_NAME/bin/activate
```
To install forkan run 
```bash
pip install -e .
```
within the git root.

## Concepts
### Separat network definitions
Networks for models and policies are typically defined in the `networks.py` in the model folder.
New networks must be named and added to the general if ... else within the file and will then be selectable
via the `network` parameter of the model class.

### Config Manager
In order to avoid passing all hyperparameter via command line and the only documentation of which values were
used being the shell history, it is possible to define configuration files that store hyperparameters
for the model, datasets and environments. 

The `forkan.common.ConfigManager` takes care of loading configured models along with datasets or policies 
with environment. It takes as argument a list of configuration names or file names containing configurations.

Training multiple configs can look like this:
```python
from forkan.common.config_manager import ConfigManager

cm = ConfigManager(config_names=['vae-breakout', 'vae-dsprites'])
cm.exec()
```

## Models

* Autoencoder
* Variational Autoencoder (with configurable beta)

## RL Algorithms

* Deep Q Networks (DQN)