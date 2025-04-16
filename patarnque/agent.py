import settings
import NN
import torch

import numpy as np

class Agent:
	def __init__(self,state_size, action_size):
		self.epsilon = 1.0
		self.action_size = action_size
		self.state_size = state_size
		self.qnet =NN.QNet(state_size, action_size).to(settings.device)
		self.qnet_target = NN.QNet(state_size, action_size).to(settings.device)

	def select_action(self, state):
		if np.random.rand() > self.epsilon:
			action = np.random.choice(self.action_size)
		else:
			self.qnet.eval()
			state = torch.FloatTensor(state).to(settings.device)
			q_values = self.qnet(state)
			action = q_values.argmax().item()
		return action
	
	def update(self, state, action, reward, next_state, done):
		self.qnet.update(state, action, reward, next_state, done, self.qnet_target)

	def update_target_network(self):
		self.qnet_target.load_state_dict(self.qnet.state_dict())

	def save_model(self,place):
		torch.save(self.qnet.state_dict(),place + 'model')