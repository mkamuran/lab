import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import torch
import settings
import parameter 
import numpy as np
import torch.optim as optim

class ReplayBuffer:
	def __init__(self, buffer_size, batch_size):
		self.buffer = deque(maxlen=buffer_size)
		self.batch_size = batch_size
	def add(self, state, action, reward, next_state, done):
		data = (state, action, reward, next_state, done)
		self.buffer.append(data)
	def __len__(self):
		return len(self.buffer)
	def get_batch(self):
		data = random.sample(self.buffer, self.batch_size)

		# 状態、次状態のテンソルをスタックする
		state = torch.stack([x[0] for x in data]).to(settings.device)
		action = torch.tensor([x[1] for x in data], dtype=torch.long, device=settings.device).unsqueeze(1)  # 変更
		reward = torch.tensor([x[2] for x in data], dtype=torch.float32, device=settings.device)
		next_state = torch.stack([x[3] for x in data]).to(settings.device)
		done = torch.tensor([x[4] for x in data], dtype=torch.float32, device=settings.device)

		return state, action, reward, next_state, done





class QNet(nn.Module):
	def __init__(self,state_size, action_size):
		super().__init__()
		self.l1 = nn.Linear(state_size, parameter.hidden1_in)
		self.l2 = nn.Linear(parameter.hidden1_in, parameter.hidden1_out)
		self.l3 = nn.Linear(parameter.hidden1_out, action_size)
		torch.nn.init.kaiming_normal_(self.l1.weight)
		torch.nn.init.kaiming_normal_(self.l2.weight)
		torch.nn.init.kaiming_normal_(self.l3.weight)
		self.buffer = ReplayBuffer(parameter.buffer_size, parameter.batch_size)
		self.optimizer = optim.Adam(self.parameters(), lr=parameter.lr)
	def forward(self, x):
		x = x.to(settings.device)
		x = F.relu(self.l1(x))
		x  = F.relu(self.l2(x))
		x = self.l3(x)
		return x
	
	def update(self,state,action,reward,next_state,done,qnet_target):
		state = torch.FloatTensor(state).to(settings.device)
		action = torch.LongTensor([action]).to(settings.device)
		reward = torch.FloatTensor([reward]).to(settings.device)
		next_state = torch.FloatTensor(next_state).to(settings.device)
		done = torch.FloatTensor([done]).to(settings.device)

		self.buffer.add(state, action, reward, next_state, done)
		if len(self.buffer) < parameter.batch_size:
			return
		states, actions, rewards, next_states, done = self.buffer.get_batch()
		actions = actions.long()
		qscore = self(states).gather(1, actions).squeeze(1)
		qnet_target.eval()
		with torch.no_grad():
			next_qscore = qnet_target(next_states).max(dim=1, keepdim=True)[0].squeeze(1)
		target = rewards + (1-done) * parameter.gamma * next_qscore


		loss = F.mse_loss(qscore, target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
			