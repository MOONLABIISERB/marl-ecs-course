import numpy as np
from typing import Dict, List, Tuple
from astar import AStarPlanner
from env import Action


class RolloutPlanner:
    def __init__(self, env, num_rollouts=30, depth=15, exploration_prob=0.2):
        self.env = env
        self.num_rollouts = num_rollouts
        self.depth = depth
        self.exploration_prob = exploration_prob
        self.astar = AStarPlanner(env.grid_size)

    def do_rollout(self) -> Dict[int, Action]:
        """Perform rollouts to find the best joint action for all agents."""
        best_outcome = float("inf")
        best_first_actions = None

        for _ in range(self.num_rollouts):
            # Make a copy of the environment for simulation
            env_copy = self.env.get_state_copy()

            # Choose first actions (will be returned if this rollout is best)
            first_actions = {}
            for agent_id in range(env_copy.num_agents):
                if agent_id not in env_copy.agents_done:
                    if np.random.random() < self.exploration_prob:
                        # Sometimes choose random action for exploration
                        valid_actions = env_copy.get_valid_actions(agent_id)
                        first_actions[agent_id] = np.random.choice(valid_actions)
                    else:
                        # Use A* for pathfinding
                        first_actions[agent_id] = self.astar.plan_action(agent_id, env_copy)
                else:
                    first_actions[agent_id] = Action.STAY

            max_steps = 0
            done = False
            total_reward = 0

            # Simulate for depth steps
            for step in range(self.depth):
                if step == 0:
                    actions = first_actions
                else:
                    # For subsequent steps, use A* with some randomness
                    actions = {}
                    for agent_id in range(env_copy.num_agents):
                        if agent_id not in env_copy.agents_done:
                            if np.random.random() < self.exploration_prob:
                                valid_actions = env_copy.get_valid_actions(agent_id)
                                actions[agent_id] = np.random.choice(valid_actions)
                            else:
                                actions[agent_id] = self.astar.plan_action(agent_id, env_copy)
                        else:
                            actions[agent_id] = Action.STAY

                # Take step in simulation
                next_states, reward, done, info = env_copy.step(actions)
                total_reward += reward
                max_steps = step + 1

                if done:
                    break

            # Calculate outcome score
            outcome = max_steps
            if not done:
                # Add estimate of remaining steps using A* path lengths
                remaining_steps = []
                for agent_id in range(env_copy.num_agents):
                    if agent_id not in env_copy.agents_done:
                        path = self.astar.find_path(env_copy.agents_pos[agent_id], env_copy.goals[agent_id], env_copy.obstacles)
                        if path:
                            remaining_steps.append(len(path) - 1)
                        else:
                            remaining_steps.append(self.depth)  # No path found, penalize heavily

                outcome = max_steps + (max(remaining_steps) if remaining_steps else 0)

            # Update best actions if this rollout found a better solution
            if outcome < best_outcome:
                best_outcome = outcome
                best_first_actions = first_actions

        return best_first_actions
