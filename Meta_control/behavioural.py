#We need to learn fb function with this module
from policy_network import *
import gym
from transition_learn import *

# theta_to_phi_psi.py

import torch
import torch.nn as nn
from policy_network import PolicyNetwork
from transition_learn import TransitionModel
from augmented_network import AugmentedNetwork

class ExpressiveThetaToPhiPsiMapping(nn.Module):
    def __init__(self, policy_network, transition_model, augmented_network, hidden_dim=512):
        super(ExpressiveThetaToPhiPsiMapping, self).__init__()
        # Store references to the networks
        self.policy_network = policy_network
        self.transition_model = transition_model
        self.augmented_network = augmented_network

        # Flatten the parameters for transition and augmented networks
        self.phi_params = nn.Parameter(torch.cat([p.view(-1) for p in transition_model.parameters()]))
        self.psi_params = nn.Parameter(torch.cat([p.view(-1) for p in augmented_network.parameters()]))
        
        # Total parameters for transition and augmented networks
        self.total_phi_psi_dim = self.phi_params.shape[0] + self.psi_params.shape[0]
        
        # A more expressive mapping from policy network's parameters theta to the concatenation of phi and psi
        # Add more hidden layers and non-linearities for a more complex mapping
        self.fc_theta_to_phi_psi = nn.Sequential(
            nn.Linear(sum(p.numel() for p in policy_network.parameters()), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_phi_psi_dim)
        )

    def forward(self):
        # Get all the parameters of the policy network
        theta_params = torch.cat([p.view(-1) for p in self.policy_network.parameters()])
        
        # Map policy network's parameters theta to the concatenation of phi and psi
        phi_psi_output = self.fc_theta_to_phi_psi(theta_params)
        
        # Split the output into phi (transition network params) and psi (augmented network params)
        phi_output = phi_psi_output[:self.phi_params.shape[0]]
        psi_output = phi_psi_output[self.phi_params.shape[0]:]
        
        return phi_output, psi_output

# Example usage
def initialize_networks_with_env(env_name='HalfCheetah-v1'):
    # Initialize the Gym environment
    env = gym.make(env_name)
    
    # Get the state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize the policy network
    policy_net = PolicyNetwork(env)
    
    # Initialize the transition model
    transition_model = TransitionModel(env)
    
    # Initialize the augmented network
    augmented_net = AugmentedNetwork(env, transition_model)

    # Initialize the expressive theta-to-[phi, psi] mapping network
    theta_to_phi_psi_net = ExpressiveThetaToPhiPsiMapping(policy_net, transition_model, augmented_net)

    return policy_net, transition_model, augmented_net, theta_to_phi_psi_net

# Test
if __name__ == "__main__":
    policy_net, transition_model, augmented_net, theta_to_phi_psi_net = initialize_networks_with_env()
    print()
    # Test the expressive theta-to-[phi, psi] mapping
    phi_output, psi_output = theta_to_phi_psi_net()
    # Calculate and print the total dimension of theta (flattened)
    total_theta_dim = sum(p.numel() for p in policy_net.parameters())

    print(f"Total dimension of theta (flattened): {total_theta_dim}")

    print(f"Output phi (transition network params): {phi_output.shape}")
    print(f"Output psi (augmented network params): {psi_output.shape}")
