import matplotlib.pyplot as plt
import pickle

def load_training_logs(log_file):
    """
    Load training logs from a pickle file.
    """
    with open(log_file, 'rb') as f:
        logs = pickle.load(f)
    return logs

logs = load_training_logs('training_logs.pkl')


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


def plot_metrics(episodes, rewards_history, steps_history):
    """
    Plot training metrics: rewards and steps over episodes.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(episodes, rewards_history, label='Total Rewards', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Total Rewards Over Episodes')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(episodes, steps_history, label='Total Steps', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Total Steps Over Episodes')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

plot_metrics(episodes, rewards_history, steps_history)
