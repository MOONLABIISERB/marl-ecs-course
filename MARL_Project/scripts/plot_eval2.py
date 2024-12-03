import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os

if __name__ == "__main__":
    seeds = [9]
    iqn_data_dir = "training_data/training_2024-11-24-23-51-15"  # Directory for IQN evaluations
    IQN2_data_dir = "training_data/training_2024-11-26-22-09-57"  # Directory for IQN2 evaluations
    colors = {"IQN": "b", "IQN2": "g"}

    fig, (ax_rewards, ax_success_rate) = plt.subplots(1, 2, figsize=(16, 6))

    # Function to process evaluation data
    def process_eval_data(data_dir, seeds):
        all_rewards, all_success_rates = [], []

        for seed in seeds:
            seed_dir = os.path.join(data_dir, "seed_" + str(seed))
            eval_data = np.load(os.path.join(seed_dir, "evaluations.npz"), allow_pickle=True)

            timesteps = np.array(eval_data['timesteps'], dtype=np.float64)
            rewards = np.mean(eval_data['rewards'], axis=1)
            success_rates = []

            for i in range(len(eval_data['timesteps'])):
                successes = eval_data['successes'][i]
                success_rates.append(np.sum(successes) / len(successes))

            all_rewards.append(rewards.tolist())
            all_success_rates.append(success_rates)

        return timesteps, all_rewards, all_success_rates

    # Process IQN data
    iqn_timesteps, iqn_rewards, iqn_success_rates = process_eval_data(iqn_data_dir, seeds)

    # Process IQN2 data
    IQN2_timesteps, IQN2_rewards, IQN2_success_rates = process_eval_data(IQN2_data_dir, seeds)

    # Plotting helper function
    def plot_data(ax, timesteps, rewards, success_rates, label, color):
        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0) / np.sqrt(np.shape(rewards)[0])
        success_rates_mean = np.mean(success_rates, axis=0)
        success_rates_std = np.std(success_rates, axis=0)

        ax[0].plot(timesteps, rewards_mean, linewidth=3, label=label, color=color)
        ax[0].fill_between(
            timesteps, rewards_mean + rewards_std, rewards_mean - rewards_std, 
            alpha=0.2, color=color
        )

        ax[1].plot(timesteps, success_rates_mean, linewidth=3, label=label, color=color)
        ax[1].fill_between(
            timesteps, success_rates_mean + success_rates_std, success_rates_mean - success_rates_std, 
            alpha=0.2, color=color
        )

    # Plot IQN and IQN2 data
    plot_data([ax_rewards, ax_success_rate], iqn_timesteps, iqn_rewards, iqn_success_rates, "IQN", colors["IQN"])
    plot_data([ax_rewards, ax_success_rate], IQN2_timesteps, IQN2_rewards, IQN2_success_rates, "IQN2", colors["IQN2"])

    # Configure plots
    mpl.rcParams["font.size"] = 25
    for ax in [ax_rewards, ax_success_rate]:
        [x.set_linewidth(1.5) for x in ax.spines.values()]
        ax.tick_params(axis="x", labelsize=22)
        ax.tick_params(axis="y", labelsize=22)
    
    ax_rewards.xaxis.set_ticks(np.arange(0, iqn_timesteps[-1] + 1, 1000000))
    ax_rewards.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e-5:.0f}'))
    ax_rewards.set_xlabel("Timestep(x10^5)", fontsize=25)
    ax_rewards.set_title("Cumulative Reward", fontsize=25, fontweight='bold')
    ax_rewards.legend()

    ax_success_rate.xaxis.set_ticks(np.arange(0, iqn_timesteps[-1] + 1, 1000000))
    ax_success_rate.set_xlabel("Timestep(x10^5)", fontsize=25)
    ax_success_rate.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
    ax_success_rate.set_title("Success Rate", fontsize=25, fontweight='bold')
    ax_success_rate.legend()

    fig.tight_layout()
    fig.savefig("learning_curves_comparison_IQN_IQN2.png")
    plt.show()
