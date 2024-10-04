import argparse
from itertools import count
from copy import deepcopy
import time
import gym
import scipy.optimize
import random

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from half_cheetah_v4 import *

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v4", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# env = gym.make(args.env_name)
env = HalfCheetahEnv(goal_vel=0.6)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
cost_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch, cost_batch, i_episode):
    costs = torch.Tensor(batch.cost)
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))
    values_cost = cost_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    returns_cost = torch.Tensor(actions.size(0),1)
    deltas_cost = torch.Tensor(actions.size(0),1)
    advantages_cost = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]
    ##
    prev_return_cost = 0
    prev_value_cost = 0
    prev_advantage_cost = 0
    for i in reversed(range(costs.size(0))):
        returns_cost[i] = costs[i] + args.gamma * prev_return_cost * masks[i]
        deltas_cost[i] = costs[i] + args.gamma * prev_value_cost * masks[i] - values_cost.data[i]
        advantages_cost[i] = deltas_cost[i] + args.gamma * args.tau * prev_advantage_cost * masks[i]

        prev_return_cost = returns_cost[i, 0]
        prev_value_cost = values_cost.data[i, 0]
        prev_advantage_cost = advantages_cost[i, 0]
    
    targets_cost = Variable(returns_cost)
    ##
    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_cost_loss(flat_params):
        set_flat_params_to(cost_net, torch.Tensor(flat_params))
        for param in cost_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        costs_ = cost_net(Variable(states))

        cost_loss = (costs_ - targets_cost).pow(2).mean()

        # weight decay
        for param in cost_net.parameters():
            cost_loss += param.pow(2).sum() * args.l2_reg
        cost_loss.backward()
        return (cost_loss.data.double().numpy(), get_flat_grad_from(cost_net).data.double().numpy())

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    if cost_batch >= -0.3 or i_episode <= 40:
        print("ok")
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
        set_flat_params_to(value_net, torch.Tensor(flat_params))

        advantages = (advantages - advantages.mean()) / advantages.std()

        action_means, action_log_stds, action_stds = policy_net(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
            else:
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()


        def get_kl():
            mean1, log_std1, std1 = policy_net(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

    else:
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_cost_loss, get_flat_params_from(cost_net).double().numpy(), maxiter=25)
        set_flat_params_to(cost_net, torch.Tensor(flat_params))

        advantages_cost = (advantages_cost - advantages_cost.mean()) / advantages_cost.std()

        action_means, action_log_stds, action_stds = policy_net(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_cost_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
            else:
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages_cost) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()


        def get_kl():
            mean1, log_std1, std1 = policy_net(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        # trpo_step(policy_net, get_cost_loss, get_kl, args.max_kl, args.damping)
        trpo_step(policy_net, get_cost_loss, get_kl, 0.05, args.damping)
    

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

EPISODE_LENGTH = 1000
EPISODE_PER_BATCH = 16
alpha = 0.02
prev_policy_net = deepcopy(policy_net)
# prev_value_net = deepcopy(value_net)
prev_cost_net = deepcopy(cost_net)

state_datas = []
velocity_datas = []
pos_datas = []
reward_datas = []
goal_vels = []
goal_vels.append(0.6)
for i_episode in count(1):
    if i_episode%20==0:
        temp_state_dict = prev_policy_net.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = temp_state_dict[key] + alpha * (policy_net.state_dict()[key] - temp_state_dict[key])
        policy_net.load_state_dict(temp_state_dict)

        # temp_state_dict = prev_value_net.state_dict()
        # for key in temp_state_dict:
        #     temp_state_dict[key] = temp_state_dict[key] + alpha * (value_net.state_dict()[key] - temp_state_dict[key])
        # value_net.load_state_dict(temp_state_dict)
        value_net = Value(num_inputs)

        temp_state_dict = prev_cost_net.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = temp_state_dict[key] + alpha * (cost_net.state_dict()[key] - temp_state_dict[key])
        cost_net.load_state_dict(temp_state_dict)

        prev_policy_net = deepcopy(policy_net)
        # prev_value_net = deepcopy(value_net)
        prev_cost_net = deepcopy(cost_net)

        goal_vel = random.uniform(0.1, 1.5)
        print(goal_vel)
        goal_vels.append(goal_vel)
        # np.save("data/metasrl/goal_vel.npy", np.array(goal_vels))
        env = HalfCheetahEnv(goal_vel=goal_vel)

        
        # np.save("data/metasrl/state.npy", np.array(state_datas))
        # np.save("data/metasrl/velocity.npy", np.array(velocity_datas))
        # np.save("data/metasrl/pos.npy", np.array(pos_datas))
        # np.save("data/metasrl/reward.npy", np.array(reward_datas))

    elif i_episode==19:
        prev_policy_net = deepcopy(policy_net)
        # prev_value_net = deepcopy(value_net)
        prev_cost_net = deepcopy(cost_net)


    memory = Memory()

    num_steps = 0
    reward_batch = 0
    cost_step = 0
    num_episodes = 0
    state = 0
    tic = time.perf_counter()
    state_data = []
    velocity_data = []
    pos_data = []
    reward_data = []
    while num_steps < EPISODE_LENGTH*EPISODE_PER_BATCH:
        state, info = env.reset()
        state = running_state(state)
        reward_sum = 0
        cost_sum = 0
        for t in range(EPISODE_LENGTH): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            cost = info["cost"]
            reward_sum += info["reward_l1"]
            cost_sum += info["cost_l1"]

            state_data.append(state)
            velocity_data.append(info["x_velocity"])
            pos_data.append(info["x_position"])
            reward_data.append(info["reward_l1"])
            next_state = running_state(next_state)

            mask = 1
            if t==EPISODE_LENGTH-1:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward, cost)
            # if args.render:
            #     env.render()
            if done or truncated:
                break

            state = next_state
        num_steps += EPISODE_LENGTH
        num_episodes += 1
        reward_batch += reward_sum
        cost_step += cost_sum
    # reward_batch /= num_episodes
    state_datas.append(state_data)
    velocity_datas.append(velocity_data)
    pos_datas.append(pos_data)
    reward_datas.append(reward_data)
    batch = memory.sample()
    update_params(batch, cost_step/num_steps, i_episode)

    if i_episode % args.log_interval == 0:
        print(f'Episode {i_episode}\tAverage reward {reward_batch/num_steps:.2f}\t Average cost {-cost_step/num_steps:.2f}')
