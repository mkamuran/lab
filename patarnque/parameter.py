#main.py
map_n = 4
sync_interval = 10
episodes = 500
max_steps = 1000
#NN
batch_size = 32
buffer_size = 10000
lr = 0.001 #初期学習率
gamma = 0.99 #Q関数の時間割引率
hidden1_in = 64
hidden1_out = 64
