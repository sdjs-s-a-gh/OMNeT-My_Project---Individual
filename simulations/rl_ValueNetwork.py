class ValueNetwork(nn.Module):
    """
        A Class representing a 64-in-64-out Feed Forward Neural Network that will act as the 'critic'
        of the PPO resource allocator. This network will calculate the expected value for the current
        state.
    """
    def __init__(self, state_space_dimensions):
        """
            Parameters:
                state_space_dimensions: the dimensions of the state space from the environment, which
                    is to be used as the input dimension.
                    At the minute, this should be 5. However, as it may change, I have generalised
                    this function to avoid hard-coding any value.

                
            Return:
                None
        """
        super(ValueNetwork, self).__init__()
        
        self.layer1 = nn.Linear(state_space_dimensions, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        
    def forward(self, state):
        """
            Completes a forward pass on the Actor/Policy neural network.
            
            Parameters:
                state: The state to use as an input.
            
            Return:
                mean: The expected value of the current state.
        """
        # Tanh can be substituted for ReLu.
        activation1 = torch.tanh(self.layer1(state))
        activation2 = torch.tanh(self.layer2(activation1))        
        mean = self.layer3(activation2)
        
        return mean