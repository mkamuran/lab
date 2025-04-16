from environment import GameMap
from agent import Agent
import random as rand
import parameter

map_n=parameter.map_n
state_size = map_n * map_n
action_size = map_n * 2
episodes = parameter.episodes
sync_interval = parameter.sync_interval
max_steps = parameter.max_steps
env=GameMap(map_n)
agent=Agent(state_size,action_size)

for episode in range(episodes+1):
	total_reward = 0
	state = env.reset()
	done = False
	agent.epsilon = 0.01 + 0.9 / (1.0+0.5*episode)
	for step in range(max_steps):
		action = agent.select_action(state)
		next_state, reward, done = env.step(action)
		agent.update(state, action, reward, next_state, done)
		state = next_state
		total_reward += reward
		if done:
			break
	if episode % sync_interval == 0:
		agent.update_target_network()
	print(f"Episode {episode}, Total Reward: {total_reward}")
agent.save_model("./model/",map_n)
