from google.colab import drive
drive.mount('/content/drive')
import os

checkpoint_path = '/content/drive/MyDrive/rdl_project'  
os.makedirs(checkpoint_path, exist_ok=True)
!pip install torch torchvision timm tqdm

import torch
import torch.nn as nn
import timm
import random
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import logging

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, 100)
checkpoint = torch.load('/content/drive/MyDrive/rdl_project/vit_cifar100_epoch_5.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Define the test set transformations and loader
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])
test_dataset = datasets.CIFAR100(root='data', train=False, transform=transform_test, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
train_dataset = datasets.CIFAR100(root='data', train=True, transform=transform_test, download=True)
train_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
import random
import pickle
from statistics import mean
from copy import deepcopy

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Agents Definition ###
class PPOAgent(nn.Module):
    def __init__(self, state_dim, num_tokens):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, num_tokens),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

### Environment ###
class MultiAgentTokenPruningEnv:
    def __init__(self, model, device, target_prune_ratio):
        self.model = model
        self.device = device
        self.target_prune_ratio = target_prune_ratio

    def reset(self, embeddings, attention_scores, similarity_scores, labels):
        self.embeddings = embeddings.to(self.device)
        self.attention_scores = attention_scores.to(self.device)
        self.similarity_scores = similarity_scores.to(self.device)
        self.labels = labels.to(self.device)
        self.batch_size, self.num_tokens = attention_scores.shape
        self.flattened_embeddings = self.embeddings.view(self.batch_size, self.num_tokens * 192)
        self.selected_tokens = [
            torch.arange(self.num_tokens).to(self.device) for _ in range(self.batch_size)
        ]

        return self.flattened_embeddings, self.flattened_embeddings

    def _get_states(self):
        # Flatten the embeddings for all tokens
        batch_size, num_tokens, embed_dim = self.embeddings.shape  # Assuming self.embeddings exists
        flattened_embeddings = self.embeddings.view(batch_size, num_tokens * embed_dim)
        return flattened_embeddings

    def step(self, actions_attention, actions_similarity, is_eval=False):
        for i in range(self.batch_size):
            combined_actions = actions_attention[i] | actions_similarity[i]      # replaced and with or
            selected_indices = (combined_actions == 1).nonzero(as_tuple=False).squeeze(1)

            if 0 not in selected_indices:
                selected_indices = torch.cat((torch.tensor([0], device=self.device), selected_indices))

            if len(selected_indices) == 1:
                selected_indices = torch.cat((selected_indices, torch.tensor([1], device=self.device)))

            self.selected_tokens[i] = self.selected_tokens[i][selected_indices]

        pruned_images = self._prune_images()

        # Compute separate rewards for each agent
        if is_eval:
          return
        else:
          rewards_attention, accuracy = self._calculate_attention_rewards(pruned_images) if not is_eval else torch.zeros(self.batch_size, device=self.device)
          rewards_similarity = self._calculate_similarity_rewards(pruned_images) if not is_eval else torch.zeros(self.batch_size, device=self.device)

          return rewards_attention, rewards_similarity, accuracy


    def _prune_images(self):
        pruned_embeddings = []
        for i in range(self.batch_size):
            pruned_indices = self.selected_tokens[i]
            pruned_embeddings.append(self.embeddings[i, pruned_indices])
        return pruned_embeddings

    def _calculate_attention_rewards(self, pruned_images):
        with torch.no_grad():
            cls_outputs = []
            for x in pruned_images:
                x = x.clone().detach().unsqueeze(0)
                for block in self.model.blocks:
                    x = block(x)
                x = self.model.norm(x)
                cls_outputs.append(x[:, 0])

            cls_outputs = torch.cat(cls_outputs, dim=0)
            outputs = self.model.head(cls_outputs)

        # Compute classification loss
        criterion = torch.nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
        classification_loss = criterion(outputs, self.labels)
        # beta = 100
        # Reward calculation
        rewards = torch.empty(self.batch_size, device=self.device)
        num_tokens_selected = 0
        for i in range(self.batch_size):
            num_tokens_selected += len(self.selected_tokens[i])
        desired_num_tokens = int(self.num_tokens * (1 - self.target_prune_ratio))
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == self.labels).float()
        beta = 1
        #     # Determine reward based on the number of selected tokens
        if num_tokens_selected <= desired_num_tokens:
          rewards[i] = beta * correct[i] / desired_num_tokens
        else:
          rewards[i] = beta * correct[i] / num_tokens_selected



        current_prune_ratios = torch.tensor(
            [
                len(self.selected_tokens[i]) / self.num_tokens
                for i in range(self.batch_size)
            ],
            device=self.device,
        )

        # Calculate deviation from the target prune ratio
        deviations = torch.abs(current_prune_ratios - (1 - self.target_prune_ratio))

        # Apply heavy penalty for deviations beyond ±0.05, no penalty otherwise
        penalty_mask = deviations > 0.05
        pruning_penalties = torch.zeros_like(deviations, device=self.device)
        pruning_penalties[penalty_mask] = deviations[penalty_mask]  # Heavy penalty multiplier

        '''rewards = (
            200 * correct
            - 600 * pruning_penalties
        )'''
        accuracy = len(correct[correct == 1])/len(correct)

        # print("Accuracy :", len(correct[correct == 1])/len(correct))
        # print("Reward :", rewards.mean())
        # print("avg Num Tokens :", num_tokens_selected/157)

        return rewards, accuracy

    def _calculate_similarity_rewards(self, pruned_images):
        with torch.no_grad():
            # Compute similarity priorities
            similarity_priorities = torch.tensor(
                [
                    self.similarity_scores[i, self.selected_tokens[i]].mean().item()
                    for i in range(self.batch_size)
                ],
                device=self.device,
            )

            # Compute classification loss (e.g., CrossEntropyLoss)
            cls_outputs = []
            for x in pruned_images:
                x = x.clone().detach().unsqueeze(0)
                for block in self.model.blocks:
                    x = block(x)
                x = self.model.norm(x)
                cls_outputs.append(x[:, 0])

            cls_outputs = torch.cat(cls_outputs, dim=0)
            outputs = self.model.head(cls_outputs)
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == self.labels).float()
            # Classification loss using ground truth labels
            classification_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            classification_losses = classification_loss_fn(outputs, self.labels)
            num_tokens_selected = 0
            for i in range(self.batch_size):
              num_tokens_selected += len(self.selected_tokens[i])
        # Compute rewards by dividing classification loss by similarity priorities
        # Add a small epsilon to similarity priorities to prevent division by zero
        epsilon = 1e-6
        alpha = 1
        rewards = alpha *correct / (similarity_priorities + epsilon)
        #print("Accuracy :", len(correct[correct == 1])/len(correct))
        #print("Reward :", rewards.mean())
        #print("avg Num Tokens :", num_tokens_selected/157)
        return rewards

### Multi-Agent Training ###
def train_multi_agent_ppo(
    agent_attention, agent_similarity, model, train_loader, env_device, num_epochs, gamma=0.99, epsilon=0.2, target_prune_ratio=0.5
):
    old_agent_attention = deepcopy(agent_attention)
    old_agent_similarity = deepcopy(agent_similarity)

    optimizer_attention_actor = optim.Adam(agent_attention.actor.parameters(), lr=3e-4)
    optimizer_attention_critic = optim.Adam(agent_attention.critic.parameters(), lr=3e-4)

    optimizer_similarity_actor = optim.Adam(agent_similarity.actor.parameters(), lr=3e-4)
    optimizer_similarity_critic = optim.Adam(agent_similarity.critic.parameters(), lr=3e-4)

    epoch_return_attn = []
    epoch_accuracy = []
    epoch_return_sim = []
    epoch_ratio_attn = []
    epoch_ratio_prune = []
    epoch_attn_policy_loss = []
    epoch_attn_value_loss = []
    epoch_sim_policy_loss = []
    epoch_sim_value_loss = []
    for epoch in range(num_epochs):
        batch_accuracy = []
        batch_ratio_attn = []
        batch_ratio_prune = []
        batch_attn_policy_loss = []
        batch_attn_value_loss = []
        batch_sim_policy_loss = []
        batch_sim_value_loss = []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(env_device), labels.to(env_device)

            # Preprocess input and get attention/similarity scores
            with torch.no_grad():
                x = model.patch_embed(images)
                batch_size = x.shape[0]

                cls_token = model.cls_token.expand(batch_size, -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                x = x + model.pos_embed

                attention_scores = model.blocks[0].attn(x).mean(dim=-1)
                x_normalized = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
                similarity_matrix = torch.matmul(x_normalized, x_normalized.transpose(1, 2))
                similarity_scores = similarity_matrix.mean(dim=-1)

            env = MultiAgentTokenPruningEnv(model, env_device, target_prune_ratio)
            states_attention, states_similarity = env.reset(x, attention_scores, similarity_scores, labels)

            # Get actions and log probabilities from current agents
            action_probs_attention, values_attention = agent_attention(states_attention)
            action_probs_similarity, values_similarity = agent_similarity(states_similarity)

            actions_attention = (action_probs_attention > 0.5).int()
            actions_similarity = (action_probs_similarity > 0.5).int()

            # Get rewards
            rewards_attn, rewards_sim, acc = env.step(actions_attention, actions_similarity)
            batch_accuracy.append(acc)

            total_tokens_kept = sum(len(env.selected_tokens[i]) for i in range(batch_size))
            total_tokens = batch_size * env.num_tokens
            batch_ratio_prune.append(total_tokens_kept / total_tokens)


            # Compute log probabilities from old agents
            with torch.no_grad():
                old_action_probs_attention, _ = old_agent_attention(states_attention)
                old_action_probs_similarity, _ = old_agent_similarity(states_similarity)

                log_probs_old_attention = torch.log(old_action_probs_attention + 1e-8)
                log_probs_old_similarity = torch.log(old_action_probs_similarity + 1e-8)

            # Update old agents after completing an epoch
            old_agent_attention.load_state_dict(agent_attention.state_dict())
            old_agent_similarity.load_state_dict(agent_similarity.state_dict())

            # Compute advantages
            advantages_attention = rewards_attn - values_attention
            advantages_similarity = rewards_sim - values_similarity

            # Optimize the agents
            attn_policy_loss, attn_value_loss, ratio_attn = optimize_agent(
                optimizer_attention_actor, optimizer_attention_critic, action_probs_attention, values_attention, advantages_attention, rewards_attn, epsilon, log_probs_old_attention
            )

            sim_policy_loss, sim_value_loss, ratio_sim = optimize_agent(
                optimizer_similarity_actor, optimizer_similarity_critic, action_probs_similarity, values_similarity, advantages_similarity, rewards_sim, epsilon, log_probs_old_similarity
            )


            avg_return_attention = rewards_attn.clone().detach().mean().item()
            avg_return_similarity = rewards_sim.clone().detach().mean().item()

            batch_ratio_attn.append(ratio_attn)
            # batch_ratio_prune.append(ratio_sim)
            batch_attn_policy_loss.append(attn_policy_loss)
            batch_attn_value_loss.append(attn_value_loss)
            batch_sim_policy_loss.append(sim_policy_loss)
            batch_sim_value_loss.append(sim_value_loss)


        epoch_accuracy.append(batch_accuracy)
        epoch_return_attn.append(avg_return_attention)
        epoch_return_sim.append(avg_return_similarity)
        epoch_ratio_attn.append(batch_ratio_attn)
        epoch_ratio_prune.append(batch_ratio_prune)
        epoch_attn_policy_loss.append(mean(batch_attn_policy_loss))
        epoch_attn_value_loss.append(mean(batch_attn_value_loss))
        epoch_sim_policy_loss.append(mean(batch_sim_policy_loss))
        epoch_sim_value_loss.append(mean(batch_sim_value_loss))

        print(f"\nAvg Accuracy {mean(epoch_accuracy[epoch])} , Avg Return (Attention): {avg_return_attention:.4f}, Avg Return (Similarity): {avg_return_similarity:.4f} , Actor Attention Loss: {epoch_attn_policy_loss[epoch]:.4f} , Critic Attention Loss: {epoch_attn_value_loss[epoch]:.4f}, Actor Similarity Loss: {epoch_sim_policy_loss[epoch]:.4f} , Critic Similarity Loss: {epoch_sim_value_loss[epoch]:.4f}")

    return epoch_accuracy, epoch_ratio_prune, epoch_ratio_attn, epoch_return_attn, epoch_return_sim, epoch_attn_policy_loss, epoch_attn_value_loss, epoch_sim_policy_loss, epoch_sim_value_loss




def optimize_agent(optimizer_actor, optimizer_critic, action_probs, values, advantages, returns, epsilon, log_probs_old):
    log_probs = torch.log(action_probs + 1e-8)
    ratios = torch.exp(log_probs - log_probs_old).mean(dim=1)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    entropy_loss = -torch.mean(action_probs * log_probs)

    policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy_loss
    value_loss = (returns - values).pow(2).mean()

    optimizer_actor.zero_grad()
    policy_loss.backward(retain_graph=True)
    optimizer_actor.step()

    optimizer_critic.zero_grad()
    value_loss.backward()
    optimizer_critic.step()

    return policy_loss.item(), value_loss.item(), ratios.mean().item()

### Initialize Model, Agents, and Train ###
# Load your ViT model and dataset
model.to(device)

state_dim = 197 * 192 # Number of tokens in ViT
num_tokens = 197

agent_attention = PPOAgent(state_dim, num_tokens).to(device)
agent_similarity = PPOAgent(state_dim, num_tokens).to(device)



num_epochs = 5
target_prune_ratio = 0.5

epoch_accuracy, epoch_ratio_prune, epoch_ratio_attn, epoch_return_attn, epoch_return_sim, epoch_attn_policy_loss, epoch_attn_value_loss, epoch_sim_policy_loss, epoch_sim_value_loss = train_multi_agent_ppo(
    agent_attention, agent_similarity, model, train_loader, device, num_epochs, gamma=0.99, epsilon=0.1, target_prune_ratio=target_prune_ratio
)


def evaluate_multi_agent(agents, model, test_loader, device, target_prune_ratio):
    """
    Evaluate a multi-agent system for token pruning on a Vision Transformer.

    Parameters:
        agents (tuple): Tuple containing the attention agent and similarity agent.
        model (nn.Module): The Vision Transformer model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run evaluation on.
        target_prune_ratio (float): Desired pruning ratio.

    Returns:
        test_accuracy (float): Classification accuracy after pruning.
        avg_tokens_kept (float): Average fraction of tokens kept after pruning.
    """
    # Unpack agents
    attention_agent, similarity_agent = agents

    # Set models to evaluation mode
    model.eval()
    attention_agent.eval()
    similarity_agent.eval()

    correct = 0
    total = 0
    total_tokens_kept = 0
    total_tokens = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating (Accuracy: 0.00%)")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Extract embeddings and attention scores
            x = model.patch_embed(images)
            batch_size = x.shape[0]
            cls_token = model.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + model.pos_embed
            attention_scores = model.blocks[0].attn(x).mean(dim=-1)
            similarity_scores = 1 - torch.cdist(x, x, p=2).mean(dim=-1)
            # Initialize environment
            env = MultiAgentTokenPruningEnv(model, device, target_prune_ratio)
            states_attention, states_similarity = env.reset(x, attention_scores,similarity_scores, labels)

            # Obtain actions from both agents
            action_probs_attention, _ = attention_agent(states_attention)
            action_probs_similarity, _ = similarity_agent(states_similarity)

            actions_attention = (action_probs_attention > 0.5).int()
            actions_similarity = (action_probs_similarity > 0.5).int()

            # Perform token pruning with combined actions
            combined_rewards = env.step(actions_attention, actions_similarity, is_eval=True)

            # Classify pruned embeddings
            pruned_images = env._prune_images()
            cls_outputs = []
            for x in pruned_images:
                x = torch.tensor(x).unsqueeze(0)
                for block in model.blocks:
                    x = block(x)  # Add batch dimension
                x = model.norm(x)
                cls_outputs.append(x[:, 0])  # Extract CLS token output

            cls_outputs = torch.cat(cls_outputs, dim=0)  # Combine outputs for all samples
            outputs = model.head(cls_outputs)

            # Compute predictions and update metrics
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update pruning statistics
            total_tokens_kept += sum(len(env.selected_tokens[i]) for i in range(batch_size))
            total_tokens += batch_size * env.num_tokens

            # Update progress bar with current accuracy
            current_accuracy = correct / total
            progress_bar.set_description(f"Evaluating (Accuracy: {current_accuracy * 100:.2f}%)")
            progress_bar.set_description(f"Evaluating (Tokens kept: {total_tokens_kept / total_tokens * 100:.2f}%)")


    test_accuracy = correct / total
    avg_tokens_kept = total_tokens_kept / total_tokens

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Average Tokens Kept: {avg_tokens_kept * 100:.2f}% (Target: {100 * (1 - target_prune_ratio):.2f}%)")

    return test_accuracy, avg_tokens_kept

test_accuracy, avg_tokens_kept = evaluate_multi_agent(
    (agent_attention, agent_similarity),
    model,
    test_loader,
    device,
    target_prune_ratio=0.5
)





def save_actor_critic_weights_and_losses(agent_attention, agent_similarity, epoch_return_attn, epoch_return_sim, epoch_attn_policy_loss, epoch_attn_value_loss, epoch_sim_policy_loss, epoch_sim_value_loss, save_dir, epochs):
    """Saves the actor and critic weights of the PPO agent."""
    actor_attn_path = f"{save_dir}/actor_attn_weights_{epochs}.pth"
    critic_attn_path = f"{save_dir}/critic_attn_weights_{epochs}.pth"
    actor_sim_path = f"{save_dir}/actor_sim_weights_{epochs}.pth"
    critic_sim_path = f"{save_dir}/critic_sim_weights_{epochs}.pth"


    # Save weights
    torch.save(agent_attention.actor.state_dict(), actor_attn_path)
    torch.save(agent_attention.critic.state_dict(), critic_attn_path)
    torch.save(agent_similarity.actor.state_dict(), actor_sim_path)
    torch.save(agent_similarity.critic.state_dict(), critic_sim_path)

    # Save losses
    with open(f"{save_dir}/epoch_return_attn_{epochs}.pkl", "wb") as f:
        pickle.dump(epoch_return_attn, f)
    with open(f"{save_dir}/epoch_return_sim_{epochs}.pkl", "wb") as f:
        pickle.dump(epoch_return_sim, f)
    with open(f"{save_dir}/epoch_attn_policy_loss_{epochs}.pkl", "wb") as f:
        pickle.dump(epoch_attn_policy_loss, f)
    with open(f"{save_dir}/epoch_attn_value_loss_{epochs}.pkl", "wb") as f:
        pickle.dump(epoch_attn_value_loss, f)
    with open(f"{save_dir}/epoch_sim_policy_loss_{epochs}.pkl", "wb") as f:
        pickle.dump(epoch_sim_policy_loss, f)
    with open(f"{save_dir}/epoch_sim_value_loss_{epochs}.pkl", "wb") as f:
        pickle.dump(epoch_sim_value_loss, f)

    print(f"Actor and Critic weights and losses saved to {save_dir}")

save_dir = '/content/drive/MyDrive/rdl_project'
epochs = num_epochs
save_actor_critic_weights_and_losses(agent_attention, agent_similarity, epoch_return_attn, epoch_return_sim, epoch_attn_policy_loss, epoch_attn_value_loss, epoch_sim_policy_loss, epoch_sim_value_loss, save_dir, epochs)
