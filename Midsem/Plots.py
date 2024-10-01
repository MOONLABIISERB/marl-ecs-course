import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

k = 10000

# Define a function for moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


df = pd.read_csv(f'Midsem/output_k_{k}.csv')

ep_rets = df['ep_rets'].tolist()
avg_losses = df['avg_losses'].tolist()

# Get maximum and minimum cumulative rewards
max_reward = max(ep_rets)
min_reward = min(ep_rets)
max_index = ep_rets.index(max_reward)
min_index = ep_rets.index(min_reward)

# Smooth the data using a window size
window_size = 50
smoothed_ep_rets = moving_average(ep_rets, window_size=window_size)

# # Plot original vs smoothed
# plt.figure(figsize=(12, 4))
# plt.plot(ep_rets, label='Original Cumulative Reward', alpha=0.5)
# plt.plot(range(window_size - 1, len(ep_rets)), smoothed_ep_rets, label='Smoothed Cumulative Reward', color='orange')
# plt.xlabel('Episode')
# plt.ylabel('Cumulative Reward')
# plt.title(f'Cumulative Reward per Episode (Smoothed) with k ={k}')
# # Add text annotations for max and min values
# plt.text(max_index, max_reward, f'Max: {max_reward:.2f}', ha='right', va='bottom', color='green', fontsize=10)
# plt.text(min_index + 1000, min_reward, f'Min: {min_reward:.2f}', ha='right', va='top', color='red', fontsize=10)
# plt.legend()
# plt.show()


# # Plot average TD error (loss)
# plt.figure(figsize=(12, 4))
# plt.plot(avg_losses, label='Average TD Error (Loss)', color='green')
# plt.xlabel('Episode')
# plt.ylabel('Average Loss')
# plt.title(f'Average Loss (TD Error) per Episode with k = {k}')

# plt.legend()

# plt.tight_layout()
# plt.show()

# Create a figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# First plot: Cumulative reward per episode (original and smoothed)
axes[0].plot(ep_rets, label='Original Cumulative Reward', alpha=0.5)
axes[0].plot(range(window_size - 1, len(ep_rets)), smoothed_ep_rets, label='Smoothed Cumulative Reward', color='orange')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Cumulative Reward')
axes[0].set_title(f'Cumulative Reward per Episode with k ={k}')


# Add text annotations for max and min values
axes[0].text(max_index, max_reward, f'Max: {max_reward:.2f}', ha='right', va='bottom', color='green', fontsize=10)

# Add horizontal offset for the min reward text by shifting the x-coordinate by 2
axes[0].text(min_index + 2000, min_reward, f'Min: {min_reward:.2f}', ha='right', va='top', color='red', fontsize=10)


axes[0].legend()

# Second plot: Average TD error (loss) per episode
axes[1].plot(avg_losses, label='Average TD Error (Loss)', color='green')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Average Loss')
axes[1].set_title(f'Average Loss (TD Error) per Episode with k = {k}')
axes[1].legend()

# Adjust layout
plt.tight_layout()
plt.show()