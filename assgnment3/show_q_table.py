import pickle
import matplotlib.pyplot as plt

# Load the pickle file
file_path = 'q_tables_env.pkl'  # Update this path if needed
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Function to plot Q-values for a single agent
def plot_agent_q_values(agent_name, agent_data):
    plt.figure(figsize=(12, 6))
    for state, q_values in agent_data.items():
        actions = range(len(q_values))  # Action indices
        plt.plot(actions, q_values, marker='o', label=f"{state}")

    plt.title(f"Q-values for {agent_name}")
    plt.xlabel("Action Index")
    plt.ylabel("Q-value")
    plt.legend(title="States", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Plot Q-values for each agent
for agent, agent_data in data.items():
    print(f"Plotting Q-values for {agent}...")
    plot_agent_q_values(agent, agent_data)
