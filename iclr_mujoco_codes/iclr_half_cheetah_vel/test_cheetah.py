import gym

# Create the HalfCheetah-v4 environment
env = gym.make('HalfCheetah-v1')

# Reset the environment to start
state = env.reset()

# # Example loop: Run the simulation for 1000 steps
# for _ in range(1000):
#     # Render the environment (optional, can slow down performance)
#     env.render()

#     # Take a random action
#     action = env.action_space.sample()

#     # Step the environment (this takes the action)
#     next_state, reward, done, info = env.step(action)

#     # If the episode is done (HalfCheetah has fallen), reset the environment
#     if done:
#         state = env.reset()

# Close the environment
env.close()
