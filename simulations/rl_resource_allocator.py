import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn

from rl_PolicyNetwork import PolicyNetwork
from rl_ValueNetwork import ValueNetwork

class RLResourceAllocator():
    def __init__(self, state_space_dimensions, action_space_dimensions, learning_rate=0.005, gamma=0.95, clip_parameter=0.2):
        # Instantiate the Actor (Policy) and Critic (Value) Neural Networks.        
        self.policy_network = PolicyNetwork(state_space_dimensions, action_space_dimensions)
        self.value_network = ValueNetwork(state_space_dimensions)
        
        self.policy_optimiser = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimiser = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Set Hyperparameters
        self.gamma = gamma
        self.clip_parameter = clip_parameter
        torch.manual_seed(1)
        
        self.times_batched = 0
        self.times_updated = 0
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probabilities = []
        self.batch_rewards = []        
        
        print("Python: RL Agent has been initialised")
    
    def select_action(self, state):
        """
            The brains of the allocation. What OMNeT will directly connect with.            
            
            Parameters:
                state: The state of the current simulation environment, including:
                    1. Required CPU cycles to process the input task.
                    2. Communication latency of that task.
                    3. Resource (CPU) Utilisation.
                    4. Queue length.
                    5. Total combined CPU cycles from the queue.                    
                
            Return:
                action: The CPU allocation for the input task.
                log_probability: The log probability for the input task.
                
        """
        # Convert the state to a tensor.
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Query the Policy/Actor network for a mean action to take.
        mean, std = self.policy_network(state_tensor)
        distribution = Normal(mean, std)
        
        # Sample an action from the distribution.
        raw_action = distribution.sample()
        
        # Convert the action to a scale of [-1,1] to use later.
        action = torch.tanh(raw_action)
        
        # Calculate the raw log probablity.
        raw_log_probability = distribution.log_prob(raw_action).sum()

        # Apply a correction for the log probability using the 'Jacobian Correction'.
        correction = torch.log(1 - action.pow(2) + 1e-6)
        
        # Correction B: for the linear scaling from width 2 to width 1
        # Because we compress the range, log_prob increases by log(2)
        scale_correction = torch.log(torch.tensor(0.5)) # which is -log(2)
        
        # Calculate the proper log probability
        log_probability = (raw_log_probability - correction - scale_correction).sum(dim=-1, keepdim=True)
        
        # Scale the action from [-1, 1] to [0,1]
        scaled_action = (action + 1) / 2
        

        
        #assert action > 0 and action <= 1 
        #action = torch.clamp(action, 1e-6, 1.0)  # replaces your assert and the == 0 check
        
        #print(f"Action: {scaled_action}")
        # Calculate the log probability for that action.
        #log_probability = distribution.log_prob(action).sum()

        
        #print(f"Times Batched: {self.times_batched}")
        if len(self.batch_states) >= 512: # Was 1024            
            self.update()
            self.times_updated += 1
            print(f"Times Updated: {self.times_updated}")
        
        return scaled_action.item(), log_probability.item()
        
    def update(self):
        # Get information from the batched experiences.
        batch_states = torch.tensor(self.batch_states, dtype=torch.float32)
        batch_actions = torch.tensor(self.batch_actions, dtype=torch.float32)
        old_log_probabilities = torch.tensor(self.batch_log_probabilities, dtype=torch.float32)
        #batch_rewards = torch.tensor(self.batch_rewards, dtype=torch.float32)
        
        # Compute Rewards to go? Don't know if this is correct.
        rewards_to_go = torch.tensor(self.compute_rewards_to_go(self.batch_rewards), dtype=torch.float32)
        
        # Normalise Rewards to go
        #rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-8)
        
        # Calculate advantages.
        value = self.value_network(batch_states)
        advantages = rewards_to_go - value.detach()
        
        # Normalising the advantages
        #advantages = (advantages - advantages.mean()) / advantages.std()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        advantages = advantages.detach()
        
        for epoch in range(4): # was 4
            # Calculate the new log probabilities. TODO: Should the value not also be in here as well in addition to being outside?
            current_log_probabilities, raw_entropy = self.policy_network.evaluate(batch_states, batch_actions)
            
            # Alter the entropy to account for squashing the action.
            entropy = raw_entropy - torch.log(torch.tensor(2.0))
            
            # Calculate the ratio.
            policy_ratio = torch.exp(current_log_probabilities - old_log_probabilities)
            
            # Calculate the surrogate losses.
            surr1 = policy_ratio * advantages
            surr2 = torch.clamp(policy_ratio, 1 - self.clip_parameter, 1 + self.clip_parameter) * advantages
            
            # Counteract Adam minimising the function.
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                
            self.policy_optimiser.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.policy_optimiser.step()
                
        values = self.value_network(batch_states).squeeze()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
        value_loss = nn.MSELoss()(values, rewards_to_go)
            
        self.value_optimiser.zero_grad()
        value_loss.backward()
        self.value_optimiser.step()
        
        print("Average reward:", sum(self.batch_rewards)/len(self.batch_rewards))
        print("Reward min/max:", min(self.batch_rewards), max(self.batch_rewards))
        print("Advantage mean:", advantages.mean().item())
        print("Advantage std:", advantages.std().item())
        print("Policy loss:", policy_loss.item())
        print("Value loss:", value_loss.item())
        print("---- PPO Update Complete ----")
        
        # Clear the batch
        self.clear_batch()
    
    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []
        discounted_reward = 0
        
        for reward in reversed(batch_rewards):          
            discounted_reward = reward + discounted_reward * self.gamma
            batch_rewards_to_go.insert(0, discounted_reward)
        
        print(f"Rewards-to-go: {batch_rewards_to_go}")
        return batch_rewards_to_go
        
    def add_to_batch(self, state, action, log_probability, reward):
        """Called from OMNeT. Stores the transition/experience from the last action."""
        self.times_batched += 1
        self.batch_states.append(state)
        self.batch_actions.append(action)
        self.batch_log_probabilities.append(log_probability)
        self.batch_rewards.append(reward)
    
    def clear_batch(self):
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probabilities = []
        self.batch_rewards = []
        