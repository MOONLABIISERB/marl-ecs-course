import torch as T
import numpy as np
from network import CriticNetwork, ActorNetwork
from buffer import PPOMemory

class Agent:
    def __init__(self ,gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.agent_name1 = 'P1'
        self.agent_name2 = 'P2'
        self.actor1 = ActorNetwork(alpha,name=self.agent_name1)
        self.critic1 = CriticNetwork(alpha,name=self.agent_name1)
        self.actor2 = ActorNetwork(alpha,name=self.agent_name2)
        self.critic2 = CriticNetwork(alpha,name=self.agent_name2)
        self.memory1 = PPOMemory(batch_size)
        self.memory2 = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done,agent_idx):
        if agent_idx=='P1':
            self.memory1.store_memory(state, action, probs, vals, reward, done)
        else:    
            self.memory2.store_memory(state, action, probs, vals, reward, done)
        


    def save_models(self):
        print('... saving models ...')
        self.actor1.save_checkpoint()
        self.critic1.save_checkpoint()
        self.actor2.save_checkpoint()
        self.critic2.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor1.load_checkpoint()
        self.critic1.load_checkpoint()
        self.actor2.load_checkpoint()
        self.critic2.load_checkpoint()
    def choose_action(self, observation,agent_idx):

        if agent_idx=='P1':
            state = T.tensor(observation, dtype=T.float).to(self.actor1.device)
            state = state.unsqueeze(0)
            dist = self.actor1(state)
            value = self.critic1(state)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor2.device)
            state = state.unsqueeze(0)
            dist = self.actor2(state)
            value = self.critic2(state)

        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self,agent_idx):
        for _ in range(self.n_epochs):
            if agent_idx=='P1':
                state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory1.generate_batches()
            else:
                    state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory2.generate_batches()


            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            if agent_idx=='P1':
                advantage = T.tensor(advantage).to(self.actor1.device)
                values = T.tensor(values).to(self.actor1.device)
            else:    
                advantage = T.tensor(advantage).to(self.actor2.device)
                values = T.tensor(values).to(self.actor2.device)
            for batch in batches:

                if agent_idx=='P1':
                    states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor1.device)
                    old_probs = T.tensor(old_prob_arr[batch]).to(self.actor1.device)
                    actions = T.tensor(action_arr[batch]).to(self.actor1.device)
                    dist = self.actor1(states)
                    critic_value = self.critic1(states)
                else:
                    states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor2.device)
                    old_probs = T.tensor(old_prob_arr[batch]).to(self.actor2.device)
                    actions = T.tensor(action_arr[batch]).to(self.actor2.device)
                    dist = self.actor2(states)
                    critic_value = self.critic2(states)
                
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                if agent_idx=='P1':
                    self.actor1.optimizer.zero_grad()
                    self.critic1.optimizer.zero_grad()
                else:
                    self.actor2.optimizer.zero_grad()
                    self.critic2.optimizer.zero_grad()
                total_loss.backward()
                if agent_idx=='P1':
                    self.actor1.optimizer.step()
                    self.critic1.optimizer.step()
                else:
                    self.actor2.optimizer.step()
                    self.critic2.optimizer.step()
        if agent_idx=='P1':      
            self.memory1.clear_memory()
        else:
            self.memory2.clear_memory() 


