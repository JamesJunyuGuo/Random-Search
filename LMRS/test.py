import gym
import numpy as np

# Initialize the Humanoid-v2 environment
env = gym.make('Humanoid-v1')

# Prepare lists to store data
observations = []
actions = []
rewards = []

# Simulate the environment for 500 steps
observation = env.reset()
for step in range(500):
    env.render()
    
    # Take a random action
    action = env.action_space.sample()
    
    # Step the environment
    observation, reward, done, info = env.step(action)
    
    # Store the data for analysis
    observations.append(observation)
    actions.append(action)
    rewards.append(reward)
    
    # If the episode is done, reset the environment
    if done:
        observation = env.reset()

# Convert data to NumPy arrays
observations = np.array(observations)
actions = np.array(actions)
rewards = np.array(rewards)

# Analyze or save the collected data
print(f"Collected {len(observations)} observations")
print(f"Collected {len(actions)} actions")
print(f"Total reward collected: {np.sum(rewards)}")

# Close the environment
env.close()
