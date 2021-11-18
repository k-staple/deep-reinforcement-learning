# My solution for Unity's Tennis environment

### Plot of Rewards
This implementation solved the environment (maximum reward between the two agents each epoch - over 100 epochs of at least +0.5) in 4,200 epochs. Below is a plot of the rewards during training. 
![Solved in 4200 epochs](https://github.com/k-staple/deep-reinforcement-learning/blob/master/p3_collab-compet/Num_Epochs_Solved_In_Tennis.png "Plot of Rewards")
![Plot of Rewards](https://github.com/k-staple/deep-reinforcement-learning/blob/master/p3_collab-compet/Reward_Plot_Tennis.png "Plot of Rewards")
![Plot of Rewards](https://github.com/k-staple/deep-reinforcement-learning/blob/master/p3_collab-compet/Averaged_Reward_Plot_Tennis.png "Plot of Rewards")

### Learning Algorithm
In this project, I implemented the DDPG algorithm to solve Unity's Tennis environment. At a high-level, this involved an agent with four neural networks: a local actor and critic that measured themselves against the target actor and critic respectively. Continuous_Control.ipynb contains the code to instantiate the agent and then prompts the agent to store its interactions with the environment as experience in its replay buffer and periodically learn. Learning involves the agent sampling experiences from its replay buffer and using them to train its actors (estimates best action) and critics (estimates total reward an action will lead to given a certain state). 
The actors' and critics' neural networks (models.py) consisted of two fully connected hidden layers with relu activations followed by a fully connected output layer. The actors' output layer had a tanh activation function because the action space was -1 to 1 while the critics' output layer had no activation function. There's dropout layers after the activation functions, but the probability is 0, so dropout is not utilized.
Both tennis players were encapsulated in the one agent, affecting the size of the neural networks but not the number of them.

For the agent's hyperparameters, I used: 
- learning rate of .0002 for actors and critics
- hidden layers with 32 then 24 nodes for all four two-layer neural networks
- batch size of 128
- replay buffer size of 1e5
- gamma (discount factor) of 0.99

The models' weights from training that solved the environment are saved in local_actor.pth, local_critic.pth, target_actor.pth, and target_critic.pth.

### Ideas for Future Improvement
To further improve results, since learning is slow at the beginning but very rapid at the end, I could try a larger learning rate with learning rate decay on the actor's and critic's optimizer to hopefully learn more rapidly. Additionally, I could try a couple different dropout probabilities for the dropout layers to see if that solves the environment more quickly. 
