import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Define the neural network model for transition dynamics
class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=10):
        super(TransitionModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state_pred = self.fc3(x)
        return next_state_pred

# Replay buffer for storing transitions
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(next_states)

    def size(self):
        return len(self.buffer)

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

class TransitionModel(nn.Module):
    def __init__(self, env, hidden_dim=100):
        super(TransitionModel, self).__init__()
        self.env = env  
        state_dim = env.observation_space.shape[0]  # State (observation) space dimension
        action_dim = env.action_space.shape[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Define the layers of the neural network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, state_dim)  # Output layer predicts the change in state (Δstate)

    def forward(self, state, action):
        # Concatenate the current state and action
        x = torch.cat([state, action], dim=1)
        # Pass through fully connected layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        delta_state = self.fc4(x)  # Predict the change in state (Δstate)
        
        # Residual learning: next_state = current_state + delta_state
        next_state_pred = state + delta_state
        return next_state_pred



# Replay buffer for storing transitions
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(next_states)

    def size(self):
        return len(self.buffer)

# Efficient training function with TensorBoard logging
def train_model(env, model, optimizer, criterion, writer, buffer_size=1000, batch_size=64, episodes=1000, gamma=0.99, min_buffer_size=1000, early_stopping_threshold=1e-3):
    replay_buffer = ReplayBuffer(buffer_size)
    total_loss_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_loss = 0
        
        while not done:
            action = env.action_space.sample()  # Random action (replace with policy if needed)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.add((state, action, next_state))

            if replay_buffer.size() >= min_buffer_size:
                # Sample a random mini-batch
                states, actions, next_states = replay_buffer.sample(batch_size)

                # Convert to PyTorch tensors
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)

                # Forward pass
                next_state_preds = model(states, actions)

                # Compute loss (Mean Squared Error)
                loss = criterion(next_state_preds, next_states)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            state = next_state
        
        total_loss_history.append(total_loss)

        # Log loss to TensorBoard
        writer.add_scalar("Loss/episode", total_loss, episode)

        # Print loss for every 100 episodes
        if episode % 2 == 0:
            print(f"Episode {episode}/{episodes}, Loss: {total_loss}")
        
        # Early stopping if loss converges
        if episode > 100 and abs(total_loss_history[-1] - total_loss_history[-2]) < early_stopping_threshold:
            print(f"Stopping early at episode {episode} due to low loss change.")
            break

# Main function to setup and run the model training
def main(env_name='Walker2d-v1', log_dir='runs'):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(state_dim,action_dim)

    # Initialize the transition model
    model = TransitionModel(env)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Train the model
    train_model(env, model, optimizer, criterion, writer)

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()

