# My solution for Unity's Banana Navigator environment

### Plot of Rewards
This implementation solved the environment (average reward over 100 episodes of at least +30) in @@@!!! episodes. Below is a plot of the rewards during training. 

![Plot of Rewards](https://github.com/k-staple/deep-reinforcement-learning/blob/update_report/p1_navigation/plot_of_rewards_during_training.PNG "Plot of Rewards")

### Learning Algorithm
In this project, I implemented the DDPG algorithm to solve Unity's Reacher environment for a single agent. At a high-level, this involved an agent with four neural networks: a local actor and critic that measured themselves against the target actor and critic respectively. Continuous_Control.ipynb contains the code to instantiate the agent and then prompts the agent to store its interactions with the environment as experience in its replay buffer and periodically learn. Learning involves the agent sampling experiences from its replay buffer and using them to train its actors (estimates best action) and critics (estimates total reward an action will lead to given a certain state). 
The four neural networks use the same architecture as https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal which was inspired by the original DDPG paper and can be found in model.py. 
The actors' neural networks consist of @@@ It involves three convolutional layers and a hidden fully connected layer that each use a ReLU activation function followed by a final fully connected layer.
The critics' neural networks consists of @@@
For the agent's hyperparameters, I used: 
- learning rate of .0001 for actors and .001 for critics
- batch size of 128
- replay buffer size of 1e6
- gamma (discount factor) of 0.99
- every twenty timesteps, updated the networks (once the replay buffer had a sufficient number of samples for learning). 

The models' weights from training that solved the environment are saved in local_actor.pth, local_critic.pth, target_actor.pth, and target_critic.pth.

### Ideas for Future Improvement
To further improve results, I could try using weight decay on the critic optimizer to see if that solves the environment more quickly. Additionally, I could expand this to a multi-agent solution.
