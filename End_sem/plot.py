import pickle
import matplotlib.pyplot as plt

# Load the pickle file
with open("game_logs.pickle", "rb") as pickle_logs:
    logs = pickle.load(pickle_logs)

# Extract data
episodes = logs["episodes"]  
pursuer_rewards = logs["purser_reward"]  
evader_rewards = logs["evader_reward"] 

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(episodes, pursuer_rewards, label="Pursuer Reward", color="blue")
plt.plot(episodes, evader_rewards, label="Evaders Reward", color="orange")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode vs. Rewards")
plt.legend()
plt.grid(True)
plt.show()
