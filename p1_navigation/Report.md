# Project 1: Navigation

### Introduction

The learning algorithm used is vanilla Deep Q Learning as described in
[original paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).
Instead of an image, a state vector is used an an input to the Deep Neural Network for this project.
In the paper, the input provided to the agent are raw pixels, thus requiring the use of a convolutional network.

The Deep Neural network architecture in `model.py`:
* Fully connected layer - input: 37 (state size); output: 64
* Fully connected layer - input: 64; output 64
* Fully connected layer - input: 64; output: (action size)

Training parameters used in `Navigation.ipynb`:
* Maximum steps per episode: 1200
* Starting epsilion: 1.0
* Ending epsilion: 0.01
* Epsilion decay rate: 0.995

Parameters used in `dqn_agent.py:
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
```

## Results

The DQN agent successfully solved the environment after 664 episodes in 13 minutes 28 seconds.
This is made possible as the agent started its training from a checkpoint that saved the progress of a previous training session.

![results](trainingplot.PNG)

### Ideas for future work

1. Learning from pixels
2. Hyperparameter optimisation
3. [Double DQN](https://arxiv.org/abs/1509.06461)
4. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
5. [Dueling DQN](https://arxiv.org/abs/1511.06581)
6. [Learning from multi-step bootstrap targets (as in A3C)](https://arxiv.org/abs/1602.01783)
7. [Distributional DQN](https://arxiv.org/abs/1707.06887)
8. [Noisy DQN](https://arxiv.org/abs/1706.10295)
9. [RAINBOW](https://arxiv.org/abs/1710.02298)