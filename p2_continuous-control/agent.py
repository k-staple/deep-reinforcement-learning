import random
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from models import Actor, Critic
import queue
from collections import namedtuple, deque
import copy

class Agent():

	def __init__(self, n_states, n_actions, random_seed=3):
		# constants
		# from paper
		self.Q_DISCOUNT = .99
		self.TAU = 0.001
		self.UPDATE_EVERY = 20

		# incrementing timestep because will only learn every self.UPDATE_EVERY timesteps
		# reset beginning of each episode
		self.ts = 0	

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.n_agents = 1
		self.n_actions = n_actions

		self.local_actor = Actor(n_states, n_actions, 256, 0).to(self.device)
		self.target_actor = Actor(n_states, n_actions, 256, 0).to(self.device)
		self.local_critic = Critic(n_states, n_actions, 256, 256, 128, 0).to(self.device)
		self.target_critic = Critic(n_states, n_actions, 256, 256, 128, 0).to(self.device)
	
		self.actor_opt = Adam(self.local_actor.parameters(), lr=.0001)
		# use wgt decay for critic? 1 Udacity DDPG doesn't
		self.critic_opt = Adam(self.local_critic.parameters(), lr=.001)

		# batch_size in the paper is 64 but 128 in 2 Udacity DDPG examples
		self.batch_size = 128
		self.replay_buffer = ReplayBuffer(self.batch_size, 1000000)
		# add noise to each action
		self.noise = OUNoise(n_actions, random_seed) 

	def save(self):
		torch.save(self.local_actor.state_dict(), 'local_actor_state_dict.pt')
		torch.save(self.target_actor.state_dict(), 'target_actor_state_dict.pt')
		torch.save(self.local_critic.state_dict(), 'local_critic_state_dict.pt')
		torch.save(self.target_critic.state_dict(), 'target_critic_state_dict.pt')

		torch.save(self.local_actor, 'local_actor.pt')
		torch.save(self.target_actor, 'target_actor.pt')
		torch.save(self.local_critic, 'local_critic.pt')
		torch.save(self.target_critic, 'target_critic.pt')

	# initialize variables that need to be at the beginning of episodes
	def reset(self):
		self.noise.reset()
		self.ts = 0

	# predict next best action using local actor
	def act(self, state, add_noise=True):
		self.local_actor.eval()
		with torch.no_grad(): 
			action = self.local_actor(torch.from_numpy(state).to(self.device))
		action = action.to("cpu").detach().numpy()		
		self.local_actor.train() 
		if add_noise:
			action += self.noise.sample()
		return np.clip(action, -1, 1)
	
	# learn periodically as suggested in Udacity's benchmark implementation page to improve stability
	def step(self, state, action, reward, next_state, done):
		self.replay_buffer.add(state, action, reward, next_state, done)
		if (self.ts % self.UPDATE_EVERY == 0) and (len(self.replay_buffer) >= self.replay_buffer.sample_size):
			experiences = self.replay_buffer.sample()
			self.learn(experiences)

		self.ts += 1

	# gradually phase in local network's weights into a target network's weights	
	def soft_update(self, target_NN, local_NN):
		for local_param, target_param in zip(local_NN.parameters(), target_NN.parameters()): 
			target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
		return target_NN
		
	# train local NNs & soft update target NNs
	def learn(self, experiences):
		states, actions, rewards, next_states, dones = experiences
	
		states = states.type(torch.FloatTensor).to(self.device)
		actions = actions.type(torch.FloatTensor).to(self.device)
		rewards = rewards.type(torch.FloatTensor).to(self.device)
		next_states = next_states.type(torch.FloatTensor).to(self.device)	
		dones = dones.type(torch.FloatTensor).to(self.device)	

		######
		# train critic
		######
		pred_q = self.local_critic(states, actions)
		
		next_actions = self.target_actor(next_states)
		# Q is the sum of discounted rewards following a particular first action
		target_q = rewards + (1 - dones) * self.Q_DISCOUNT * self.target_critic(next_states, next_actions)

		critic_loss = F.mse_loss(pred_q.to("cpu"), target_q.to("cpu")) 
		self.critic_opt.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm(self.local_critic.parameters(), 1)
		self.critic_opt.step()
		
		######
		# train actor
		######
		pred_actions = self.local_actor(states)
		# critic used to critique actor: larger Q is better so minimize the negative of it
		actor_loss = -self.local_critic(states, pred_actions).to("cpu").mean()
		self.actor_opt.zero_grad()
		actor_loss.backward()
		self.actor_opt.step()
	
		######
		# soft update target NNs
		######
		self.target_critic = self.soft_update(self.target_critic, self.local_critic)
		self.target_actor = self.soft_update(self.target_actor, self.local_actor)
	
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer():
	def __init__(self, batch_size, buffer_size):
		self.sample_size = batch_size
		self.buffer_size = buffer_size

		self.q = deque(maxlen=buffer_size)
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
	def add(self, state, action, reward, next_state, done):
		"""Add an experience to memory so memory can later be sampled in order for NNs to learn"""
		self.q.append(self.experience(state, action, reward, next_state, done))
	def sample(self):
		"""Return a sample of experiences to use in as input in a NN's forward pass"""
		samples = random.sample(self.q, self.sample_size)

		# re-orgs the splits. samples is a sample of experience tuples "horizontal records" vs the alike characteristics separated out "vertical cols." Learning uses all states for example when calculating pred_actions with local_actor
		states = torch.from_numpy(np.vstack([tuple_t.state for tuple_t in samples if tuple_t is not None]))
		actions = torch.from_numpy(np.vstack([tuple_t.action for tuple_t in samples if tuple_t is not None]))
		rewards = torch.from_numpy(np.vstack([tuple_t.reward for tuple_t in samples if tuple_t is not None]))
		next_states = torch.from_numpy(np.vstack([tuple_t.state for tuple_t in samples if tuple_t is not None]))
		dones = torch.from_numpy(np.vstack([tuple_t.state for tuple_t in samples if tuple_t is not None]))

		return (states, actions, rewards, next_states, dones)
	def __len__(self):
		return len(self.q)	
