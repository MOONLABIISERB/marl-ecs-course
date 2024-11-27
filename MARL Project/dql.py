import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class DQNAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=4.5e-7, target_update=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.learn_step_counter = 0
        self.target_update = target_update  # How often to update the target network

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        self.Q_target = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                     fc1_dims=256, fc2_dims=256)

        # Initialize the target network with the same weights as Q_eval
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()  # Target network does not update during backpropagation

        # Replay buffer
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)  # Use target network for Q_next
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

        # Update the target network after every `target_update` steps
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())



# import torch as T
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np


# class DeepQNetwork(nn.Module):
#     def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
#         super(DeepQNetwork, self).__init__()
#         self.fc1 = nn.Linear(*input_dims, fc1_dims)
#         self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        
#         self.fc3 = nn.Linear(fc2_dims, n_actions)
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#         self.loss = nn.MSELoss()
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)

#     def forward(self, state):
#         x = T.relu(self.fc1(state))
#         x = T.relu(self.fc2(x))
#         actions = self.fc3(x)
#         return actions


# class DQNAgent:
#     def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
#                  max_mem_size=100000, eps_end=0.01, eps_dec=1e-5):
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.eps_min = eps_end
#         self.eps_dec = eps_dec
#         self.lr = lr
#         self.action_space = [i for i in range(n_actions)]
#         self.mem_size = max_mem_size
#         self.batch_size = batch_size
#         self.mem_cntr = 0

#         self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
#                                    fc1_dims=256, fc2_dims=256)

#         # Replay buffer
#         self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
#         self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
#         self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
#         self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

#     def store_transition(self, state, action, reward, state_, done):
#         index = self.mem_cntr % self.mem_size
#         self.state_memory[index] = state
#         self.new_state_memory[index] = state_
#         self.reward_memory[index] = reward
#         self.action_memory[index] = action
#         self.terminal_memory[index] = done

#         self.mem_cntr += 1

#     def choose_action(self, observation):
#         if np.random.random() > self.epsilon:
#             state = T.tensor([observation], dtype=T.float).to(self.Q_eval.device)
#             actions = self.Q_eval.forward(state)
#             action = T.argmax(actions).item()
#         else:
#             action = np.random.choice(self.action_space)
#         return action

#     def learn(self):
#         if self.mem_cntr < self.batch_size:
#             return

#         self.Q_eval.optimizer.zero_grad()

#         max_mem = min(self.mem_cntr, self.mem_size)
#         batch = np.random.choice(max_mem, self.batch_size, replace=False)
#         batch_index = np.arange(self.batch_size, dtype=np.int32)

#         state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
#         new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
#         reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
#         terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

#         action_batch = self.action_memory[batch]

#         q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
#         q_next = self.Q_eval.forward(new_state_batch)
#         q_next[terminal_batch] = 0.0

#         q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

#         loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
#         loss.backward()
#         self.Q_eval.optimizer.step()

#         self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)
