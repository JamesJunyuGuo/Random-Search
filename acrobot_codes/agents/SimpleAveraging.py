import numpy as np
from utils.torch_utils import use_cuda, Tensor, Variable, ValueFunctionWrapper
from utils.CRPO import *
from utils.DQNSoftmax import *
from utils.DQNRegressor import *
from acrobot import *
from copy import deepcopy


class SimpleAveraging:
    def __init__(
        self,
        input_size,
        output_size,
        alpha = 0.01,
        value_function_lr = 1,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.value_function_lr = value_function_lr
        ## Initial neural network
        self.policy = DQNSoftmax(input_size, output_size)
        self.value_function = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        self.cost_value_function_1 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        self.cost_value_function_2 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        ## Cuda identity
        # self.policy.cuda()
        # self.value_function.cuda()
        # self.cost_value_function_1.cuda()
        # self.cost_value_function_2.cuda()

        self.prev_policies = []
        self.prev_value_functions = []
        self.prev_cost_value_functions_1 = []
        self.prev_cost_value_functions_2 = []

        self.rewards_by_task = []
        self.cost_1s_by_task = []
        self.cost_2s_by_task = []


    def step(self, noise: np.array, crpo_step = 10, crpo_episodes = 10, cg_iters = 10, limit_1 = 50, limit_2 = 50, direction = 0):
        env = AcrobotEnv(noise)
        crpo = CRPO(env, self.input_size, self.output_size, self.policy, self.value_function, self.cost_value_function_1, self.cost_value_function_2,
                    cg_iters=cg_iters,
                    episodes=crpo_episodes,
                    limit_1=limit_1,
                    limit_2=limit_2,
                    direction = direction)
        rewards = []
        cost_1s = []
        cost_2s = []
        for _ in range(crpo_step):
            reward, cost_1, cost_2 = crpo.step()
            rewards.append(reward)
            cost_1s.append(cost_1)
            cost_2s.append(cost_2)
            print("Reward: {:.2f} - Cost 1: {:.2f} - Cost 2: {:.2f}".format(reward, cost_1, cost_2))
        self.rewards_by_task.append(rewards)
        self.cost_1s_by_task.append(cost_1s)
        self.cost_2s_by_task.append(cost_2s)
        
        self.prev_policies.append(deepcopy(crpo.policy))
        self.prev_value_functions.append(deepcopy(crpo.value_function))
        self.prev_cost_value_functions_1.append(deepcopy(crpo.cost_value_function_1))
        self.prev_cost_value_functions_2.append(deepcopy(crpo.cost_value_function_2))