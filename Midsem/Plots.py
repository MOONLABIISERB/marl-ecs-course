import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Define a function for moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


df = pd.read_csv('Midsem/output.csv')

ep_rets = df['ep_rets']
avg_losses = df['avg_losses']


# Smooth the data using a window size
window_size = 50
smoothed_ep_rets = moving_average(ep_rets, window_size=window_size)

# # Plot original vs smoothed
# plt.plot(ep_rets, label='Original Cumulative Reward', alpha=0.5)
# plt.plot(range(window_size - 1, len(ep_rets)), smoothed_ep_rets, label='Smoothed Cumulative Reward', color='orange')
# plt.xlabel('Episode')
# plt.ylabel('Cumulative Reward')
# plt.title('Cumulative Reward per Episode (Smoothed)')
# plt.legend()
# plt.show()


# # Plot average TD error (loss)
# plt.figure(figsize=(12, 4))
# plt.plot(avg_losses, label='Average TD Error (Loss)', color='green')
# plt.xlabel('Episode')
# plt.ylabel('Average Loss')
# plt.title('Average Loss (TD Error) per Episode')
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
axes[0].set_title('Cumulative Reward per Episode')
axes[0].legend()

# Second plot: Average TD error (loss) per episode
axes[1].plot(avg_losses, label='Average TD Error (Loss)', color='green')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Average Loss')
axes[1].set_title('Average Loss (TD Error) per Episode')
axes[1].legend()

# Adjust layout
plt.tight_layout()
plt.show()