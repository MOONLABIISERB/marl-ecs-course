import pickle
from environment import MultiAgentGymEnv
from agent import QLearningAgent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(environment, agents, total_episodes=100, max_steps=10000, log_dir="./logs"):
    """
    Conduct multi-agent reinforcement learning rollouts.

    :param environment: Multi-agent environment instance.
    :param agents: Dictionary of agent instances.
    :param total_episodes: Number of training episodes.
    :param max_steps: Maximum steps per episode.
    :return: Rewards per episode, average rewards, and learned Q-table.
    """

    writer = SummaryWriter(log_dir=log_dir)
    cumulative_rewards = {agent_id: 0 for agent_id in agents.keys()}
    episode_rewards = {agent_id: [] for agent_id in agents.keys()}
    q_table = {}

    for episode in tqdm(range(total_episodes), desc="Training Episodes"):
        state = environment.reset()
        # environment.render()
        episode_done = False
        episode_reward = {agent_id: 0 for agent_id in agents.keys()}
        step_count = 0

        for step in tqdm(range(max_steps), desc=f"Episode {episode+1}", leave=False):
            if episode_done:
                break
            actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id], q_table = agent.select_action(state, q_table)

            # Step through the environment
            next_state, rewards, episode_done, _ = environment.step(actions)

            # Update Q-values for each agent
            for agent_id, agent in agents.items():
                next_action, q_table = agent.select_action(next_state, q_table)
                episode_reward[agent_id] += rewards[agent_id]
                q_table = agent.update_q_values(
                    state, actions[agent_id], rewards[agent_id], next_state, episode_done, q_table
                )

            state = next_state
            step_count += 1
            # environment.render()

        # Log rewards
        if episode % (total_episodes // 10_000) == 0:
            print(f"Episode {episode}/{total_episodes}: Reward = {episode_reward}")

        for agent_id in agents.keys():
            writer.add_scalar(f"Agent_{agent_id}/Episode_Reward", episode_reward[agent_id], episode)
            episode_rewards[agent_id].append(episode_reward[agent_id])
            cumulative_rewards[agent_id] += episode_reward[agent_id]

        # Log cumulative reward updates to TensorBoard
        for agent_id, total_reward in cumulative_rewards.items():
            writer.add_scalar(f"Agent_{agent_id}/Cumulative_Reward", total_reward, episode)

    avg_rewards = {agent_id: total / total_episodes for agent_id, total in cumulative_rewards.items()}
    return episode_rewards, avg_rewards, q_table

def save_q_table(q_table, filename='q_table.pkl'):
    """
    Save the Q-table to a file.

    :param q_table: Dictionary containing Q-values for states and actions.
    :param filename: File path for saving the Q-table.
    """
    with open(filename, 'wb') as file:
        pickle.dump(q_table, file)
        print(f"Q-table successfully saved to {filename}")

if __name__ == "__main__":
    # Define the environment parameters
    init_positions = {
        0: (1,1),
        1: (8,1),
        2: (1,8),
        3: (8,8)
    }

    goals = {
        0: (5,8),
        1: (1,5),
        2: (8,4),
        3: (5,1)
    }
    obstacles=[
        (0, 4),
        (1, 4),
        (2, 4),
        (2, 5),
        (4, 0),
        (4, 1),
        (4, 2),
        (5, 2),
        (4, 9),
        (4, 8),
        (4, 7),
        (5, 7),
        (9, 5),
        (8, 5),
        (7, 5),
        (7, 4)
    ]

    # Training hyperparameters
    max_steps_per_episode = 10000
    num_training_episodes = 10000000
    exploration_rate = 0.1
    learning_rate = 0.2
    discount_factor = 0.9

    # Initialize environment and agents
    environment = MultiAgentGymEnv(goals=goals, init_positions=init_positions, obstacles=obstacles)
    agents = {
        agent_id: QLearningAgent(
            agent_id=agent_id,
            action_space=environment.action_space,
            exploration_rate=exploration_rate,
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )
        for agent_id in range(4)
    }

    # Run MARL training
    rewards, avg_rewards, learned_q_table = train(
        environment=environment,
        agents=agents,
        total_episodes=num_training_episodes,
        max_steps=max_steps_per_episode
    )

    print(f"Average rewards per agent over {num_training_episodes} episodes: {avg_rewards}")

    # Save the resulting Q-table
    save_q_table(learned_q_table, filename="learned_q_table.pkl")