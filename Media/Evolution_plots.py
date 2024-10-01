import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _folders
import json
import numpy as np
import sys

_folders.set_experiment_folders('_Optimization')
results_path = f'{_folders.RESULTS_PATH}/Evolution_Fourier.json'

data = json.load(open(results_path, 'r'))
print(data.keys())

rewards = np.array(data['REWARDS'])
generations = len(rewards)

max_rewards = np.max(rewards, axis=1)
print('Max Rewards per Generation:', max_rewards)
max_reward_gen = np.argmax(max_rewards)
max_reward_individual = np.argmax(rewards[max_reward_gen])
print('Max Reward Gen:', max_reward_gen, 'Max Reward Individual:', max_reward_individual)


#plot best regard and mean reward
import matplotlib.pyplot as plt


mean_rewards = np.mean(rewards, axis=1)
std_rewards = np.std(rewards, axis=1)

import _colors
palette = _colors.create_palette(2)
plt.plot(max_rewards, label='Best Individual', color=palette[0])
plt.plot(mean_rewards, label='Population Mean Reward', color=palette[1])
plt.fill_between(np.arange(generations), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

plt.xlabel('Generation')
plt.ylabel('Reward')
plt.legend()
plt.show()