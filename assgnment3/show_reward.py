import pickle
import matplotlib.pyplot as plt

# Load the training logs from the .pkl file
file_path = 'training_logs_env.pkl'  # Update this to the correct file path
with open(file_path, 'rb') as file:
    rewards = pickle.load(file)

# Check if the data is in the expected format (array)
if isinstance(rewards, (list, tuple)):
    # Plot the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward', color='blue')
    plt.title('Training Rewards Over Time')
    plt.xlabel('Training Iterations')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("The data is not in an array form. Please check the file content.")
