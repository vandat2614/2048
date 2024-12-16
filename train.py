import random
from board import Board
from agent import Agent
from buffer import ReplayMemory

NUM_EPISODE = 100
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 0.99
BATCH_SIZE = 32
TARGET_UPDATE = 5 
MAX_EPISODE_LENGTH = 100

global_step = 0
env = Board(size=4)
agent = Agent()
buffer = ReplayMemory(capacity=10000)

print(f'CHECK: {agent.device}')

for episode in range(NUM_EPISODE):
    state = env.reset()
    episode_length = 0
    while True:
        epsilon = max(EPS_END, EPS_START * (EPS_DECAY ** global_step))
        if random.random() > epsilon:
            action = agent.select_action(state)
        else:
            action = random.randint(0, 3)

        next_state, reward, done = env.step(action)
        episode_length += 1

        buffer.push(state, action, reward, next_state, done)

        state = next_state
        global_step += 1

        if len(buffer) >= BATCH_SIZE:
            transitions = buffer.sample(BATCH_SIZE)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
            agent.update_model(list(batch_state), batch_action, batch_reward, list(batch_next_state), batch_done)

        if done:
            print(f"Episode {episode + 1}: Total Reward = {env.score} Max = {max(next_state)}")
            break

    if episode % TARGET_UPDATE == 0:
        agent.update_target_network()
