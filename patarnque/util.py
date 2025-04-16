import numpy as np
from collections import deque
import copy
import random as rand

def is_goal(map):
		return all(x == 0 for x in map)

def tate(map, x,n):
	for i in range(n):
		idx = i * n + x
		map[idx] = 1 - map[idx]
	for i in range(n // 2):
		top = i * n + x
		bottom = (n - 1 - i) * n + x
		map[top], map[bottom] = map[bottom], map[top]

def yoko(map, y,n):
	for i in range(n):
		idx = y * n + i
		map[idx] = 1 - map[idx]
	for i in range(n // 2):
		left = y * n + i
		right = y * n + (n - 1 - i)
		map[left], map[right] = map[right], map[left]

def apply_action(map, command,n):
	if command < n:
		tate(map,command,n)
	else:
		yoko(map,command - n,n)

def solve(map):
	n = int(len(map)**0.5)
	visited = set()
	queue = deque()
	queue.append((copy.copy(map), []))
	visited.add(tuple(map))

	while queue:
		now_map, path = queue.popleft()
		if is_goal(now_map):
			return path
		for command in range(2 * n):
			temp= copy.copy(now_map)
			apply_action(temp,command,n)
			if tuple(temp) not in visited:
				visited.add(tuple(temp))
				queue.append((temp, path + [command]))
	return None

def is_colectans(map,path):
	temp = copy.deepcopy(map)
	for command in path:
		apply_action(temp,command)
	if is_goal(temp):
		return True
	return False

