# Project report

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, the training was done in the second version of the environment wich contains 20 identical agents, each with its own copy of the environment.

## Learning algorithm
In order to solve the environment I implemented a PPO agent according to the Proximal Policy Optimization Algorithms paper.

PPO is an actor critic model. The idea behind it, is that we limit how far we can change our policy in each iteration through the KL-divergence, where the KL-divergence measures the difference between two data distributions p and q which are our old and new policies.

## Parameters and hyperparameters
### Neural networks

The model consists of 2 neural networks - actor and critic.

#### Actor network
- 3 fully connected layers
- 33 input nodes [observation vector size], 4 output nodes [action vector size], 512 hidden nodes in each layer
- ReLU activations, tanh on last layer

#### Critic network
- 3 fully connected layers
- 33 input nodes [observation vector size], 1 output nodes, 512 hidden nodes in each layer
- ReLU activations, no activation on last layer
- 
### Main hyperparameters
- Discount rate - `0.99`
- Tau - `0.95`
- Gradient clip - `0.2`
- Learning rate - `3e-4`
- Epsilon - `0.1`
- Beta - `0.01`

## Plot of Rewards
Environment solved in 134 episodes!	Average Score: 30.02  
![image](https://user-images.githubusercontent.com/65574771/211800714-d3cc6d1b-ec37-475b-87ff-63a4538ec6be.png)

## Ideas for Future Work
In the future id like to compare the PPO algorithms preformance to others like DDPG or A3C. 
Id also like to experiment with a bigger network and more tuning to the hyperparameters to achive better results.
