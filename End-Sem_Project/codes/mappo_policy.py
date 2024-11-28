import torch 
import torch.nn as nn

class MAPPOPolicy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(MAPPOPolicy, self).__init__()
        self.obs_shape = obs_shape
        self.action_space = action_space
        input_size = obs_shape[0] * obs_shape[1]

        self.network = nn.Sequential(
            nn.Linear(input_size, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def forward(self, obs):
        obs_flat = obs.view(obs.size(0), -1).reshape(1,-1)
        return self.network(obs_flat)

    def get_action(self, obs):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)
