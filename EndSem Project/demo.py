import torch
from MADQN import MADQN
# Assuming MADQN class and DQNetwork are already defined in your script

def load_madqn_model(checkpoint_path, device):
    """
    Load the model and optimizer states from a checkpoint.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize MADQN model with the same configuration as before
    n_agents = checkpoint['n_agents']
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    
    # Create an instance of MADQN and load saved state_dicts
    madqn = MADQN(n_agents, state_dim, action_dim, writer=None)
    
    # Load model weights
    for agent_id in range(n_agents):
        madqn.q_networks[agent_id].load_state_dict(checkpoint['models'][f'agent_{agent_id}'])
        madqn.target_networks[agent_id].load_state_dict(checkpoint['models'][f'agent_{agent_id}'])
        madqn.optimizers[agent_id].load_state_dict(checkpoint['optimizers'][f'agent_{agent_id}'])
    
    # Restore training state
    madqn.epsilon = checkpoint['epsilon']
    madqn.training_step = checkpoint['training_step']
    
    print(f"Checkpoint loaded from episode {checkpoint['episode']}")
    
    return madqn

# Usage:
checkpoint_path = 'checkpoints/best_model/checkpoint_episode_8.pth'  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
madqn = load_madqn_model(checkpoint_path, device)
