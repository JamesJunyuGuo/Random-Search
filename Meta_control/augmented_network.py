import torch
import torch.nn as nn
import gym
from transition_learn import TransitionModel  # Import TransitionModel from transition_learn.py

# Define the augmented network g(s, psi) = f(s, 0) + h(s, psi)
class AugmentedNetwork(nn.Module):
    def __init__(self, env, transition_model, hidden_dim=10):
        super(AugmentedNetwork, self).__init__()
        # Store the environment inside the class
        self.env = env
        
        # Get the state and action dimensions from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Use the original transition model f(s, a)
        self.transition_model = transition_model

        # Define the new neural network h(s, psi) where psi are the learnable parameters
        self.fc1_h = nn.Linear(self.state_dim, hidden_dim)
      
        self.fc2_h = nn.Linear(hidden_dim, self.state_dim)  # Output for state prediction

    def forward(self, state):
        # First part: f(s, 0), where action = 0 (with appropriate action dimension)
        zero_action = torch.zeros(state.shape[0], self.action_dim).to(state.device)  # Create zero action with correct action dimension
        f_s_0 = self.transition_model(state, zero_action)  # Get f(s, 0)
        # print("Print action", zero_action)

        # Second part: h(s, psi)
        x_h = torch.relu(self.fc1_h(state))
        x_h = torch.relu(self.fc2_h(x_h))
        # h_s_psi = self.fc3_h(x_h)

        # Combine f(s, 0) + h(s, psi)
        g_s_psi = f_s_0 + x_h
        return g_s_psi

# # Function to initialize the augmented network using the environment
# def initialize_augmented_network_with_env(env_name='HalfCheetah-v1'):
#     # Initialize the Gym environment
#     env = gym.make(env_name)
    
#     # Initialize the original transition model using the state and action dimensions from the environment
#     original_model = TransitionModel(env.observation_space.shape[0], env.action_space.shape[0])
    
#     # Initialize the augmented network using the environment and the transition model
#     augmented_model = AugmentedNetwork(env, original_model)

#     return augmented_model, env

# # Example usage
# augmented_model, env = initialize_augmented_network_with_env()

# # Print the structure of the new augmented network
# print(augmented_model)

# # Test the augmented network with a sample state
# state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
# output = augmented_model(state)

# print(f"Output from the augmented network: {output}")
