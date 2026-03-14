import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn

from rl_PolicyNetwork import PolicyNetwork
from rl_ValueNetwork import ValueNetwork

class PPO:
    def __init__(self, state_space_dimensions, action_space_dimensions) -> None:
        self.state_space_dimensions = state_space_dimensions
        self.action_space_dimensions = action_space_dimensions
        
        # PPO Algorithm Step 1: Initialising the Policy (Actor) and Value (Critic) networks.
        self.policy_network = PolicyNetwork(self.state_space_dimensions, self.action_space_dimensions)
        self.value_network = ValueNetwork(self.state_space_dimensions)
        
        # Default Hyperparameter Values
        self.batch_size = 512           # Number of timesteps per episode.
        self.updates_per_episode = 5    # Number of times to update the policy/actor and value/critic networks per episode.
        self.learning_rate = 0.005      # Learning Rate of the policy and value optimisers.
        self.gamma = 0.95               # Discount factor to be used for cal
        self.clip_parameter = 0.2       # Value to clip the ratio when calculating surrogate 2.
        
        
        
        # Optimisers for more stable convergence
        self.policy_optimiser = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimiser = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        
        # Set a seed value for reproducible results.
        torch.manual_seed(1)
        
        self.batch_actions = []
        self.batch_states = []
        self.batch_log_probabilities = []
        self.batch_rewards = []
        self.all_episode_rewards = []
        
    def get_action(self, state):
        """
            Returns the percentage of CPU that the resource allocator should give to the incoming task. This
            subroutine is to be called from OMNeT.

            Parameters:
                state: The current state of the simulation at the time a task is set to be allocated some CPU resources.

            Returns:
                action (float): The percentage of CPU to be used to compute the input task given as a ratio.

                log_probability (float): The log probability of the action taken, which is just the confidence the network
                has of it being successful / maximising the reward.
        """
        # Query the Policy/Actor network for a mean action and the standard deviation.
        mean, std = self.policy_network(state)

        # Create a distribution with the mean action and the standard deviation.
        distribution = Normal(mean, std)

        # Sample an action from the distribution.
        action = distribution.sample()

        # Calculate the log probability for that action to be successful.
        log_probability = distribution.log_prob(action)

        return action.detach(), log_probability.detach()     
   
    def add_trajectory(self, action, log_probability, new_state, reward):
        """
            Adds the input trajectory to the current batch.

            This subroutine is to be called from OMNeT upon completion of a timestep, 
            which for me is when a specific task has entered and finished being executed.

            Parameters:
                action (float)
                log_probability (float)
                new_state (float)
                reward (float)
        """     
        self.batch_actions.append(action)
        self.batch_log_probabilities.append(log_probability)
        self.batch_states.append(new_state)
        self.all_episode_rewards.append(reward)

        # PPO Algorithm Step 2: Learn for some number of iterations. In this case,
        # an interation's length = to the batch size, which itself is equal to the episode length. 
        # was - which itself is double the episode length.
        if len(self.batch_actions) % self.batch_size == 0:
            # Add the latest batch/episode rewards to the overall episodes.
            self.all_episode_rewards.append(self.batch_rewards)
            
            # Update both the policy and value networks.
            self.learn()

            # Once the episode has ended, clear the batch to prepare for the new one.
            self.clear()
    
    
    def learn(self):
        # PPO Algorithm Step 3: Collect trajectories/experiences from the most recent episode
        # and convert them into separate tensors.
        batch_actions = torch.tensor(self.batch_actions, dtype=float)
        batch_states = torch.tensor(self.batch_states, dtype=float)
        batch_log_probablities = torch.tensor(self.log_probabilities, dtype=float)
        batch_rewards = torch.tensor(self.batch_rewards, dtype=float)
        
        # PPO Algorithm Step 4: Calculate Rewards to Go
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)
        
        value = self.value_network(batch_states)
        
        # PPO Algorithm Step 5: Calculate Advantages
        advantages = batch_rewards_to_go - value.detach()
        
        # Normalise Advantages for improved stability.
        advantages = (advantages - advantages.mean() / (advantages.std() + 1e-10))
        
        for epoch in range(self.updates_per_episode):
            # Calculate the Value and Current Log probabilities for the current epoch.
            value = self.value_network(batch_states)            
            current_log_probabilities, _ = self.policy_network.evaluate(batch_states, batch_actions)
            
            # Calculate ratios for the surrogate losses.
            ratios = torch.exp(current_log_probabilities - batch_log_probablities)
            
            # Calculate Surrogate Losses.
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_parameter, 1 + self.clip_parameter) * advantages   
            
            # PPO Algorithm Step 6: Update the Policy.
            policy_loss = (-torch.min(surr1, surr2)).mean()
            
            self.policy_optimiser.zero_grad()
            policy_loss.backward(retain_graph = True)
            self.policy_optimiser.step()
            
            # PPO Algorithm Step 7: Fit Value function by regression on MSE using the
            # predicted values at the current epoch.
            value_loss = nn.MSELoss()(value, batch_rewards_to_go)
            
            self.value_optimiser.zero_grad()
            value_loss.backward()
            self.value_optimiser.step()          
            
    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []
        
        for episode_rewards in reversed(self.all_episode_rewards):
            discounted_reward = 0
            
            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                
                batch_rewards_to_go.insert(0, discounted_reward)
        
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=float)
        
        return batch_rewards_to_go


    def clear(self):
        self.batch_actions = []
        self.log_probabilities = []
        self.states = []
        self.all_episode_rewards = []
    