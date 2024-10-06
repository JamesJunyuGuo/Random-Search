import numpy as np
from agents.MetaSRL import *
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Optional app description')
# Optional argument
parser.add_argument('--train_task_count', type=int, help='An optional integer argument')
parser.add_argument('--crpo_step_count', type=int, help='An optional integer argument')
parser.add_argument('--crpo_episode_count', type=int, help='An optional integer argument')
parser.add_argument('--run', type=int, help='An optional integer argument')
args = parser.parse_args()

RUN = 1

TRAIN_TASK_COUNT = 1

CRPO_STEP_COUNT = 2
CRPO_EPISODE_COUNT = 2
CG_ITER_COUNT = 2

INPUT_SIZE = 6
OUTPUT_SIZE = 3
VARIANCE = 0.2
LIMIT_RANGE = [40, 42]

print("Finish generating the args")

metasrl = MetaSRL(INPUT_SIZE, OUTPUT_SIZE)
np.random.seed(0)
noises = np.random.normal(0.0, VARIANCE, size=(TRAIN_TASK_COUNT, 4))
limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TRAIN_TASK_COUNT))
for i in range(TRAIN_TASK_COUNT):
    print("Task #{}".format(i))
    noise = noises[i]
    limit = limits[i]

    metasrl.step(noise=noise, crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=i%2)
    plt.plot(metasrl.rewards_by_task[-1], label="Reward")
    plt.plot(metasrl.cost_1s_by_task[-1], label="Cost 1")
    plt.plot(metasrl.cost_2s_by_task[-1], label="Cost 2")
    plt.hlines(y=limit, xmin=0, xmax=CRPO_STEP_COUNT, colors="black", linestyles="--", label="Limit 1")
    plt.legend(loc="upper right")
    plt.title(f'Performance of MetaSRL on Task {i}')
    plt.xlabel("CRPO Runs")
    plt.ylabel("Performance")
    plt.savefig('./results/plot_{}.png'.format(i))
    plt.close()

    performance = np.array([metasrl.rewards_by_task, metasrl.cost_1s_by_task, metasrl.cost_2s_by_task])
    np.save(f"results/MetaSRL/performance_{i}.npy", performance)
    torch.save(metasrl.policy.state_dict(), f"results/MetaSRL/model_{i}.pth")



from utils.DQNSoftmax import *
model = DQNSoftmax(INPUT_SIZE,OUTPUT_SIZE)  # Initialize the model
model.load_state_dict(torch.load(f"results/MetaSRL/model_{i}.pth"))

    # torch.save(metasrl.policy, f"results/MetaSRL/run_{RUN}/models/model_{i}")
    # torch.save(metasrl.value_function, f"results/MetaSRL/run_{RUN}/models/value_function_{i}")
    # torch.save(metasrl.cost_value_function_1, f"results/MetaSRL/run_{RUN}/models/cost_value_function_1_{i}")
    # torch.save(metasrl.cost_value_function_2, f"results/MetaSRL/run_{RUN}/models/cost_value_function_2_{i}")

# import numpy as np
# from agents.Strawman import *
# import matplotlib.pyplot as plt
# import torch
# import argparse

# parser = argparse.ArgumentParser(description='Optional app description')
# # Optional argument
# parser.add_argument('--train_task_count', type=int, help='An optional integer argument')
# parser.add_argument('--crpo_step_count', type=int, help='An optional integer argument')
# parser.add_argument('--crpo_episode_count', type=int, help='An optional integer argument')
# parser.add_argument('--run', type=int, help='An optional integer argument')
# args = parser.parse_args()

# RUN = 1

# TRAIN_TASK_COUNT = 2

# CRPO_STEP_COUNT = 2
# CRPO_EPISODE_COUNT = 2
# CG_ITER_COUNT = 5

# INPUT_SIZE = 6
# OUTPUT_SIZE = 3
# VARIANCE = 0.2
# LIMIT_RANGE = [40, 42]


# strawman = Strawman(INPUT_SIZE, OUTPUT_SIZE)
# np.random.seed(RUN)
# noises = np.random.normal(0.0, VARIANCE, size=(TRAIN_TASK_COUNT, 4))
# limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TRAIN_TASK_COUNT))
# for i in range(TRAIN_TASK_COUNT-1, TRAIN_TASK_COUNT):
#     print("Task #{}".format(i))
#     noise = noises[i]
#     limit = limits[i]
#     strawman.step(noise=noise, crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=i%2)

#     plt.plot(strawman.rewards_by_task[-1], label="Reward")
#     plt.plot(strawman.cost_1s_by_task[-1], label="Cost 1")
#     plt.plot(strawman.cost_2s_by_task[-1], label="Cost 2")
#     plt.hlines(y=limit, xmin=0, xmax=CRPO_STEP_COUNT, colors="black", linestyles="--", label="Limit")
#     plt.legend(loc="upper right")
#     plt.title(f'Performance of Strawman on Task {i}')
#     plt.xlabel("CRPO Runs")
#     plt.ylabel("Performance")
#     plt.savefig(f'results/Strawman/run_{RUN}/plots/plot_{i}')
#     plt.close()

#     performance = np.array([strawman.rewards_by_task, strawman.cost_1s_by_task, strawman.cost_2s_by_task])
#     np.save(f"results/Strawman/run_{RUN}/performance_data/performance_{i}.npy", performance)
#     print(type(strawman.policy), type(strawman.value_function),type(strawman.cost_value_function_1),type(strawman.cost_value_function_2))

    # torch.save(strawman.policy, f"results/Strawman/model_{i}")
    # torch.save(strawman.value_function, f"results/Strawman/run_{RUN}/models/value_function_{i}")
    # torch.save(strawman.cost_value_function_1, f"results/Strawman/run_{RUN}/cost_value_function_1_{i}")
    # torch.save(strawman.cost_value_function_2, f"results/Strawman/run_{RUN}/cost_value_function_2_{i}")

# ## Test time
# print("########## Test Time ##########")
# TEST_TASK_COUNT = 10

# CRPO_STEP_COUNT = 5
# CRPO_EPISODE_COUNT = 10
# CG_ITER_COUNT = 5

# test_noises = np.random.normal(0.0, VARIANCE, size=(TEST_TASK_COUNT, 4))
# test_limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TEST_TASK_COUNT))
# for i in range(TEST_TASK_COUNT):
#     print("Test Task #{}".format(i))
#     test_noise = test_noises[i]
#     test_limit = test_limits[i]
#     strawman.evaluate(noise=test_noise, crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=test_limit, limit_2=test_limit, direction=i%2)
    
#     test_performance = np.array([strawman.test_rewards_by_task, strawman.test_cost_1s_by_task, strawman.test_cost_2s_by_task])
#     np.save(f"results/Strawman/run_{RUN}/performance_data/test_performance_{i}.npy", test_performance)