# forkan

This repository contains implementations of selected deep learning models and reinforcement algorithms.

Additional README files can be found in some model and algorithm directories that go more into detail
on the implementation of this particular method.

## Package Structure

`common` provides functions that are used in multiple subpackages

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


## Models

* Autoencoder
* Variational Autoencoder (with configurable beta)

## RL Algorithms

* Deep Q Networks (DQN)
* Advantage Actor Critic (A2C)