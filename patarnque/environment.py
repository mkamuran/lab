import numpy as np
import random
import copy
class GameMap:
	def __init__(self, n=5, black_count=3):
		self.n = n
		self.size = n * n
		self.map_data = np.zeros(self.size, dtype=int)
		black = random.sample(range(0, self.size), black_count)
		for i in black:
			self.map_data[i] = 1

	def set_map(self, map_data):
		self.map_data = np.array(map_data, dtype=int)

	def reset(self):
		self.map_data = np.zeros(self.size, dtype=int)
		black = random.sample(range(0, self.size), 3)
		for i in black:
			self.map_data[i] = 1
		return self.map_data
	
	def apply_action(self, command):
		if command < self.n:
			self.revers_y(command, self.map_data)
		else:
			self.revers_x(command - self.n, self.map_data)

	def revers_y(self, x, map_data):
		for i in range(self.n):
			idx = i * self.n + x
			map_data[idx] = 1 - map_data[idx]
		for i in range(self.n // 2):
			top = i * self.n + x
			bottom = (self.n - 1 - i) * self.n + x
			map_data[top], map_data[bottom] = map_data[bottom], map_data[top]

	def revers_x(self, y, map_data):
		for i in range(self.n):
			idx = y * self.n + i
			map_data[idx] = 1 - map_data[idx]
		for i in range(self.n // 2):
			left = y * self.n + i
			right = y * self.n + (self.n - 1 - i)
			map_data[left], map_data[right] = map_data[right], map_data[left]

	def print_map(self):
		for i in range(self.size):
			print(int(self.map_data[i]), end="")
			if i % self.n == self.n - 1:
				print()
		print()

	def is_goal(self):
		return all(x == 0 for x in self.map_data)

	def reward(self):
		if self.is_goal():
			return 1
		else:
			return -1

	def step(self, action):
		self.apply_action(action)
		reward = self.reward()
		done = self.is_goal()
		return copy.deepcopy(self.map_data), reward, done