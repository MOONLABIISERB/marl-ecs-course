import matplotlib.pyplot as plt
import pickle

def load_training_logs(log_file):
    """
    Load training logs from a pickle file.
    """
    with open(log_file, 'rb') as f:
        logs = pickle.load(f)
    return logs

logs = load_training_logs('D:/MARL/Assignment_3/Q2/training_logs.pickle')

def extract_metrics_from_logs(logs):
    """
    Extract metrics such as total rewards and steps from training logs.
    """
    episodes = []
    rewards_history = []
    steps_history = []

    for log in logs:
        episodes.append(log['episode'])
        total_rewards = sum(log['total_rewards'].values())
        total_steps = len(log['steps'])
        
        rewards_history.append(total_rewards)
        steps_history.append(total_steps)

    return episodes, rewards_history, steps_history

episodes, rewards_history, steps_history = extract_metrics_from_logs(logs)

def plot_rewards_and_steps(episodes, rewards_history, steps_history):
    """
    Plot total rewards and total steps over episodes on a single plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(episodes, rewards_history, label='Total Rewards', color='green')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewards', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    ax2 = ax1.twinx()
    ax2.plot(episodes, steps_history, label='Total Steps', color='red')
    ax2.set_ylabel('Steps', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Rewards and Steps vs Episodes')
    ax1.grid()

    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.8), fontsize=10)

    plt.tight_layout()

    plt.show()

plot_rewards_and_steps(episodes, rewards_history, steps_history)
