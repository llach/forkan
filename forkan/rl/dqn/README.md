# DQN

This directory contains the core implementation of the Deep Q Learning algorithm.

## Available Algorithm Enhancements

* Double DQN
* Dueling DQN
* Prioritorized Replay
* Reward clipping
* Gradient clipping (by norm)

## Convenience features

* Model loading & saving
* Continuation of Training
* Tensorboard visualisation

`dqn.py` contains the *DQN* class which implements the algorithm as well as the enhancements.

Networks that may be used are defined in `networks.py` and can be passed to *DQN* using the `network_type` parameter. 

## Usage
In order to use the *DQN* class, it can be either instantiated directly or the *ConfigManager* is used to load a *DQN* 
instance with hyperparameters specified in a configuration file. For more details on the use of the *ConfigManager*, 
please refer to the main README of this repository.

If `dqn.py` is executed directly, a *DQN* training on the _CartPole-v0_ environment is started.

## Paper
[Nature](https://www.nature.com/articles/nature14236.pdf)

[Original](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)