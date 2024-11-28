import torch
import torch.optim as optim
from mappo_policy import MAPPOPolicy


class MAPPOAgent:
    def __init__(self, obs_shape, action_space, lr=0.0003):
        self.policy = MAPPOPolicy(obs_shape, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def to(self, device):
        """
        Move the policy network to the specified device.
        """
        self.policy.to(device)

    def update(self, observations, actions, log_probs, rewards, next_obs, dones, gamma=0.99, clip_ratio=0.2, device="cpu"):
        """
        Update policy using PPO.
        """
        obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, next_log_probs = self.policy.get_action(next_obs_tensor)
            advantages = (rewards_tensor + gamma * next_log_probs * (1 - dones_tensor) - log_probs).detach()
        _, current_log_probs = self.policy.get_action(obs_tensor)
        ratio = torch.exp(current_log_probs - log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()
