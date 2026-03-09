import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn

from rl_PolicyNetwork import PolicyNetwork
from rl_ValueNetwork import ValueNetwork

class RLResourceAllocator():
    def __init__(self, state_space_dimensions, action_space_dimensions, learning_rate=3e-4, gamma=0.99, clip_parameter=0.2):
        # Instantiate the Actor (Policy) and Critic (Value) Neural Networks.        
        self.policy_network = PolicyNetwork(state_space_dimensions, action_space_dimensions)
        self.value_network = ValueNetwork(state_space_dimensions)
        
        self.policy_optimiser = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimiser = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Set Hyperparameters
        self.gamma = gamma
        self.clip_parameter = clip_parameter
        
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
        
        # Restrict the action to being between -1 and +1.   
        action = torch.tanh(raw_action)
        
        #print(f"Action before contraint: {action}")
        
        # Constrain the action so that it is not invalid (> 0 and <=1.
        action = (action + 1) / 2   
        #print(f"Action after contraint: {action}")   
        
        assert action > 0 and action <= 1 
        
        # Calculate the log probability for that action.
        log_probability = distribution.log_prob(raw_action).sum()
        
        #print(f"Times Batched: {self.times_batched}")
        if len(self.batch_states) >= 512: # Was 1024            
            self.update()
            self.times_updated += 1
            print(f"Times Updated: {self.times_updated}")
        
        return raw_action.item(), action.item(), log_probability.item()
        
    def update(self):
        # Get information from the batched experiences.
        batch_states = torch.tensor(self.batch_states, dtype=torch.float32)
        batch_actions = torch.tensor(self.batch_actions)
        old_log_probabilities = torch.tensor(self.batch_log_probabilities, dtype=torch.float32)
        #batch_rewards = torch.tensor(self.batch_rewards, dtype=torch.float32)
        
        # Compute Rewards to go? Don't know if this is correct
        rewards_to_go = torch.tensor(self.compute_rewards_to_go(self.batch_rewards), dtype=torch.float32)
        
        # Calculate advantages.
        value = self.value_network(batch_states)
        advantages = rewards_to_go - value
        
        # Normalising the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        advantages = advantages.detach()
        
        for epoch in range(4): # was 4
            # Calculate the new log probabilities. TODO: Should the value not also be in here as well in addition to being outside?
            current_log_probabilities, entropy = self.policy_network.evaluate(batch_states, batch_actions)
            
            # Calculate the ratio.
            ratio = torch.exp(current_log_probabilities - old_log_probabilities)
            
            # Calculate the surrogate losses.
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_parameter, 1 + self.clip_parameter) * advantages
            
            # Counteract Adam minimising the function.
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            
            self.policy_optimiser.zero_grad()
            policy_loss.backward()
            self.policy_optimiser.step()
            
        values = self.value_network(batch_states).squeeze()
        value_loss = nn.MSELoss()(values, rewards_to_go)
        
        self.value_optimiser.zero_grad()
        value_loss.backward()
        self.value_optimiser.step()
        
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
    
    def decide_allocation(self, current_load):
        print(f"Python Received load {current_load}.")
        return 1 if current_load > 0.5 else 0
    
    def allocate_resources_static(self, required_cpu_cycles):
        """
        A dummy function used just to ensure that this script can be accessed from the Simulator.
        """
        return required_cpu_cycles
    
    def allocate_resources_ppo_dummy(self, max_cpu_capacity, required_cpu_cycles, resource_utilisation):
        """
        Input:
            # CPU Cycles Required
            # Maybe deadline latency
            
        State: 
            # Network Conditions (connection quality expressed as SINR (Alex); Path loss (Chen); Packet loss, communication latency, bandwidth (Mahimalmur))
            # Queue Length (Liu, Mahimalmur)
            # Waiting time for pending tasks (Mahimalmur)
            # Resource Utilisation (Mahimalur) or Availability (Liu)
            # Current Latency of the system (which would just be the average of each task combined)
            # Current Energy Consumption of the system (which would just be the average of each task combined) (Alex, Mahimalmur)            
        
        Action/Output:
            # Allocate x number of CPU cycles to the task.
        """
        print(f"Max CPU Capacity: {max_cpu_capacity}; Required CPU Cycles: {required_cpu_cycles}; Resource Utilisation: {resource_utilisation}")
        if resource_utilisation < 0.5:
            return int(max_cpu_capacity/2)
        else:
            return required_cpu_cycles
        
    def allocate_resources(self, required_cpu_cycles, communication_latency, resource_utilisation, queue_length, total_cpu_cycles_in_queue):
        """
        args: 
        * communication_latency: Measured in milliseconds, the time delay between the task leaving the sending device and arriving on the edge server.
        """
        print(f"Required CPU Cycles: {required_cpu_cycles}; Communication Latency: {communication_latency}; Resource Utilisation: {resource_utilisation}; Queue Length: {queue_length}; CPU Cycles in Queue: {total_cpu_cycles_in_queue}")
        return required_cpu_cycles
        
        