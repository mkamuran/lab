from environment import GameMap
import settings
import torch
import random as rand
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np

def display_board(n,board):
	# 盤面を作成（ランダムに0と1を配置）

	# 描画
	fig, ax = plt.subplots(figsize=(6, 6))  # 盤面の大きさを設定
	ax.imshow(board, cmap='binary', interpolation='nearest')  # 盤面を白黒で表示

	ax.set_xticks(np.arange(n+1)-0.5, minor=True)  # x軸に目盛りを設定
	ax.set_yticks(np.arange(n+1)-0.5, minor=True)  # y軸に目盛りを設定

	# 目盛りを描画
	ax.grid(which='minor', color='black', linestyle='-', linewidth=1)  # 格子線を描画

	# 軸のラベルを非表示
	ax.set_xticks([])  
	ax.set_yticks([])

	# 盤面を表示
	plt.title(f"{n}x{n} Board")
	plt.show()



def test(n):
	black_count=rand.randrange(2,8)
	agent=Agent(n**2,n*2)
	env=GameMap(n,3)
	# 'weights_only=True' を指定して、モデルの重みのみを読み込む
	agent.qnet.load_state_dict(torch.load("./model", map_location=settings.device, weights_only=True))

	steps=0
	state=env.reset(black_count)
	done=False
	while(not done and steps <= 100):
		action=agent.select_action(state)
		env.print_map()
		print(action)
		print()
		display_board(n,state)
		state,_,done=env.step(action)
	
	if(done):
		print("ok")
	else:
		print("no")

test(3)
