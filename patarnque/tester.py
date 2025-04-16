import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # ← これを最初に！
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from environment import GameMap
import settings
import random as rand
from agent import Agent
import matplotlib.pyplot as plt
from util import solve
def display_board(n,board):
	fig, ax = plt.subplots(figsize=(6, 6))  # 盤面の大きさを設定
	ax.imshow(board, cmap='binary', interpolation='nearest')  # 盤面を白黒で表示
	ax.set_xticks(np.arange(n+1)-0.5, minor=True)  # x軸に目盛りを設定
	ax.set_yticks(np.arange(n+1)-0.5, minor=True)  # y軸に目盛りを設定
	ax.grid(which='minor', color='black', linestyle='-', linewidth=1)  # 格子線を描画
	ax.set_xticks([])  
	ax.set_yticks([])
	plt.title(f"{n}x{n} Board")
	plt.show()

def animate_boards(n, boards,best_path):
	fig, ax = plt.subplots(figsize=(6, 6))
	img = ax.imshow(boards[0], cmap='binary', interpolation='nearest')
	ax.set_xticks(np.arange(n+1) - 0.5, minor=True)
	ax.set_yticks(np.arange(n+1) - 0.5, minor=True)
	ax.set_title(f"Step {0}/{len(boards) - 1} best_path={best_path}")
	def update(frame):
		img.set_array(boards[frame])
		ax.set_title(f"Step {frame}/{len(boards) - 1} best_path={best_path}")
		return [img]

	ani = animation.FuncAnimation(fig, update, frames=len(boards), interval=500,repeat=False)
	plt.show()


def test(n):
	agent=Agent(n**2,n*2)
	env=GameMap(n)
	agent.qnet.load_state_dict(torch.load("./result/", map_location=settings.device, weights_only=True))

	steps=0
	state=env.reset()
	done=False
	boards=[]
	board=np.zeros((n,n))
	best_path=solve(state)
	while(steps <= 100):
		action=agent.select_action(state)
		for i in range(n):
			for j in range(n):
				board[i][j]=state[i*n+j]
		boards.append(board.copy())
		if done:
			break
		state,_,done=env.step(action)
		steps+=1
	animate_boards(n,boards,len(best_path))
	if(done):
		print("ok")
	else:
		print("no")

test(3)
