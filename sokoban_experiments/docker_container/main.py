import sys
import argparse
import gym
import gym_sokoban
import pickle 
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='REINFORCE with PyTorch')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
        help='discount factor (default: 0.99)')
parser.add_argument('--alpha', type=float, default=0.001, metavar='A',
        help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
        help='random seed (default: 123)')
parser.add_argument('--hidden_size', type=int, default=1_000, metavar='H',
        help='size of the hidden layer (default: 128)')
parser.add_argument('--drop_rate', type=float, default=0.6, metavar='H',
        help='rate to drop connections in hidden layer (default: 0.6)')
parser.add_argument('--render', type=bool, default=False,            
        help='render the environment, set to anything to render')
parser.add_argument('--episodes', type=int, default=10000, metavar='E',
        help='number of episodes to train (default:10000)')
parser.add_argument('--steps', type=int, default=10000, metavar='S',
        help='maximum steps per episode (default:10000)')
args = parser.parse_args()


env = gym.make("Sokoban-v0")

reward_threshold = env.spec.reward_threshold

env.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "models/policy.torch"

cumulative_reward = np.zeros(args.episodes)

class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.linear2 = nn.Linear(hidden_size, action_size)
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.softmax(x, dim=1)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        dist = self(state)
        action_dist = Categorical(dist)
        action = action_dist.sample()
        lp = action_dist.log_prob(action)
        self.log_probs.append(lp)
        return action.item()

    def learn(self, optim):
        trajectory_reward = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]: 
            trajectory_reward = r + args.gamma * trajectory_reward
            returns.insert(0,trajectory_reward)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for lp, reward in zip(self.log_probs, returns):
            policy_loss.append(-lp * reward)
        optim.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optim.step()
        self.rewards = []
        self.log_probs = []

policy = Policy(env.observation_space, env.action_space.n, args.hidden_size).to(device)
optimizer = optim.Adam(policy.parameters(), lr=args.alpha)
eps = np.finfo(np.float32).eps.item()

def main():
    running_reward = 0 
    for i in range(1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0
        for t in range(1, args.steps + 1):
            action = policy.get_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            episode_reward += reward
            policy.rewards.append(reward)
            if done:
                break
        if running_reward == 0:
            running_reward = episode_reward
        else:
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        cumulative_reward[i-1] = episode_reward
        policy.learn(optimizer)
        if i % 10 == 0:
            print("Episode {}\tTimesteps: {}\tReward: {}\tAverage reward: {:.2f}\tRunning reward: {:.2f}".format(
                i, t, episode_reward, np.mean(cumulative_reward[:i] if len(cumulative_reward) > 0 else 0), 
                running_reward))
        if running_reward > reward_threshold:
            print("Passed threshold. Episode: {} Running reward: {} Timesteps: {}".format(i, running_reward, t))
            break
    reward_file_name = "data/" + args.env + "_hl_" + str(args.hidden_size) + "_r_" + str(round(np.sum(cumulative_reward), 2)) + ".pkl"
    pickle.dump(cumulative_reward, open(reward_file_name, "wb"))
    torch.save(policy, model_path) #load to load

if __name__ == '__main__':
    main()
