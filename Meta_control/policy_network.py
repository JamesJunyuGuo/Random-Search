import torch
import torch.nn as nn
import gym
import numpy as np 
class PolicyNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        # Extract state and action dimensions from the environment
        state_dim = env.observation_space.shape[0]  # State (observation) space dimension
        action_dim = env.action_space.shape[0]  
        # Action space dimension
        self.env = env  
       
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Define the fully connected layers for the policy network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer that predicts the action
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        # Action scaling: continuous actions often have a range (e.g., [-1, 1])
        # Use Tanh activation to ensure the action stays within the valid range.
        self.action_scale = nn.Tanh()  # Tanh ensures output is within [-1, 1]

    def forward(self, state):
        # Forward pass through the network
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # Predict the action with a tanh activation to ensure output is between [-1, 1]
        action = self.action_scale(self.fc4(x))
        return action

# # Example of how to use this in a Gym environment
# def main(env_name='HalfCheetah-v1'):
#     # Create the environment
#     env = gym.make(env_name)
    
#     # Initialize the policy network using the environment
#     policy_net = PolicyNetwork(env)
    
#     # Test the policy network by passing in a random state from the environment
#     state = env.reset()  # Get initial state from the environment
#     state_tensor = torch.tensor(state, dtype=torch.float32)  # Convert state to tensor
    
#     # Get the action from the policy network
#     action = policy_net(state_tensor)
#     action  = action.detach().numpy()
#     next_state, reward, done, _ = env.step(action)
#     print(reward)
    
#     print(f"Predicted action: {action}")

# if __name__ == "__main__":
#     main()
