import numpy as np
import random
from collections import deque
import copy
from itertools import product
from collections import defaultdict
import pandas as pd
class gameMap:
	def __init__(self, n=5, black_count=3):
		self.n = n
		self.size = n * n
		self.map_data = np.zeros(self.size, dtype=int)
		black = random.sample(range(0, self.size), black_count)
		for i in black:
			self.map_data[i] = 1

	def set_map(self, map_data):
		self.map_data = np.array(map_data, dtype=int)

	def print_map(self):
		for i in range(self.size):
			print(int(self.map_data[i]), end="")
			if i % self.n == self.n - 1:
				print()
		print()

	def tate(self, x, map_data):
		for i in range(self.n):
			idx = i * self.n + x
			map_data[idx] = 1 - map_data[idx]
		for i in range(self.n // 2):
			top = i * self.n + x
			bottom = (self.n - 1 - i) * self.n + x
			map_data[top], map_data[bottom] = map_data[bottom], map_data[top]

	def yoko(self, y, map_data):
		for i in range(self.n):
			idx = y * self.n + i
			map_data[idx] = 1 - map_data[idx]
		for i in range(self.n // 2):
			left = y * self.n + i
			right = y * self.n + (self.n - 1 - i)
			map_data[left], map_data[right] = map_data[right], map_data[left]

	def apply_action(self, command):
		if command < self.n:
			self.tate(command, self.map_data)
		else:
			self.yoko(command - self.n, self.map_data)

	def play(self):
		while True:
			print("______")
			self.print_map()
			command = input(f"0〜{self.n-1}=縦, {self.n}〜{2*self.n-1}=横, {2*self.n}=終了: ")
			print()
			if not command.isdigit():
				continue
			c = int(command)
			if c >= 2 * self.n:
				print("ゲーム終了！")
				break
			self.apply_action( c)

	def is_goal(self):
		return all(x == 0 for x in self.map_data)

def solve(origin):
	n=origin.n
	visited = set()
	queue = deque()
	queue.append((copy.deepcopy(origin), []))
	visited.add(tuple(origin.map_data))

	while queue:
		now_map, path = queue.popleft()
		if now_map.is_goal():
			return path
		for command in range(2 * n):
			temp= copy.deepcopy(now_map)
			temp.apply_action(command)
			if tuple(temp.map_data) not in visited:
				visited.add(tuple(temp.map_data))
				queue.append((temp, path + [command]))
	return None

def is_colectans(map,path):
	temp = copy.deepcopy(map)
	for command in path:
		temp.apply_action(command)
	if temp.is_goal():
		return True
	return False

# 実行
if __name__ == "__main__":
	static=dict()
	static2=defaultdict(int)
	n = 3
	origin = gameMap(n)
	for map_data in product([0, 1], repeat=n*n):
		black_count = sum(map_data)
		if black_count == 0:
			continue
		origin.set_map(map_data)
		solution = solve(origin)
		if solution:
			if(not is_colectans(origin,solution)):
				print("error")
			else:
				steps = len(solution)
				static[(black_count,steps)]=static.get((black_count,steps),0)+1
				static2[(black_count,"O")] += 1
		else:
			static[(black_count, -1)] = static.get((black_count,-1),0)+1
			static2[(black_count,"X")] += 1

print(f"{n}*{n} result")
for key in sorted(static.keys()):
	print(f"黒:{key[0]} 手数:{key[1]} → 件数:{static[key]}")
for key in sorted(static2.keys()):
	print(f"黒:{key[0]} → {key[1]}件数:{static2[key]}")

df = pd.DataFrame([(k[0], k[1], v) for k, v in static.items()],columns=["black","steps","value"])
miss=df[df["steps"]==-1]["value"].sum()
clear=df[df["steps"]!=-1]["value"].sum()
print(f"miss: {miss} clear: {clear}")

