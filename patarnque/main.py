from environment import GameMap
from agent import Agent
map_n=3
state_size = map_n * map_n
action_size = map_n * 2
black_count = 3
episodes = 1000
sync_interval = 10
max_steps = 1000
env=GameMap(map_n, black_count)
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
