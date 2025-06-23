
import numpy as np
import gym
from gym import spaces

class GridWorldEnv(gym.Env):
    """
    Simple GridWorld environment for RL benchmarking.
    Supports curriculum difficulty and multi-agent extension.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, size=5, n_agents=1, curriculum_level=1):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.n_agents = n_agents
        self.curriculum_level = curriculum_level
        self.action_space = spaces.Discrete(4) 
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        self.reset()

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.size-1, self.size-1])
        return self._get_obs()

    def step(self, action):
        if action == 0: 
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1: 
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.size-1)
        elif action == 2: 
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.size-1)
        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1.0 if done else -0.01
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.agent_pos.copy()

    def render(self, mode='human'):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        print('\n'.join(' '.join(row) for row in grid))
        print()

    def reset(self):
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([self.size-1, self.size-1], dtype=np.int32)
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.int32)


try:
    import gym
    from gym import spaces
except ImportError:
 
    import numpy as np

    class spaces:
        class Discrete:
            def __init__(self, n):
                self.n = n
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

    class gym:
        Env = object

import gym
import pybullet as p
import pybullet_envs
import numpy as np

class RobotEnv(gym.Env):
    """
    PyBullet-based robotic environment for RL benchmarking.
    """
    def __init__(self, render=False):
        super(RobotEnv, self).__init__()
        self.render_mode = render
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        self.env = gym.make('InvertedPendulumBulletEnv-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        pass 

    def close(self):
        p.disconnect(self.physicsClient)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99):
        self.model = DQNNet(obs_dim, act_dim)
        self.target = DQNNet(obs_dim, act_dim)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()
        self.update_steps = 0

    def act(self, obs, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.model.net[-1].out_features)
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q = self.model(obs)
        return q.argmax().item()

    def update(self, batch):
        obs, act, rew, next_obs, done = batch
        obs = torch.FloatTensor(obs)
        act = torch.LongTensor(act)
        rew = torch.FloatTensor(rew)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done)
        q = self.model(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(next_obs).max(1)[0]
            target = rew + self.gamma * q_next * (1 - done)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % 100 == 0:
            self.target.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.model.state_dict())

    def export_torchscript(self, path):
        traced = torch.jit.trace(self.model, torch.randn(1, self.model.net[0].in_features))
        traced.save(path)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNet(obs_dim, act_dim)
        self.value = ValueNet(obs_dim)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def act(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def evaluate(self, obs, act):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(act)
        entropy = dist.entropy()
        values = self.value(obs)
        return log_probs, entropy, values

    def update(self, batch):
        obs, act, old_log_probs, returns, advs = batch
        obs = torch.FloatTensor(obs)
        act = torch.LongTensor(act)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advs = torch.FloatTensor(advs)
        for _ in range(4):
            log_probs, entropy, values = self.evaluate(obs, act)
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advs
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, path):
        torch.save({'policy': self.policy.state_dict(), 'value': self.value.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])

    def export_torchscript(self, path):
        traced = torch.jit.trace(self.policy, torch.randn(1, self.policy.net[0].in_features))
        traced.save(path)

import os
import json
import time

class Logger:
    def __init__(self, config=None):
        self.logs = []
        self.config = config or {}
        self.start_time = time.time()

    def log(self, data):
        data['time'] = time.time() - self.start_time
        self.logs.append(data)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.logs, f)

    def print_last(self):
        if self.logs:
            print(self.logs[-1])

import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, config=None):
        self.config = config or {}

    def plot_rewards(self, rewards, label='Reward', save_path=None):
        sns.set(style="darkgrid")
        plt.figure(figsize=(8,4))
        plt.plot(rewards, label=label)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_comparison(self, rewards_dict, save_path=None):
        sns.set(style="darkgrid")
        plt.figure(figsize=(8,4))
        for label, rewards in rewards_dict.items():
            plt.plot(rewards, label=label)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

import numpy as np

class ReplayBuffer:
    def __init__(self, size, obs_dim):
        self.size = size
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.cur_size = 0, 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.cur_size = min(self.cur_size + 1, self.size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.cur_size, batch_size, replace=False)
        return (self.obs_buf[idxs], self.act_buf[idxs], self.rew_buf[idxs],
                self.next_obs_buf[idxs], self.done_buf[idxs])

import argparse
import numpy as np
from devprime.envs.gridworld import GridWorldEnv
from devprime.envs.robot_env import RobotEnv
from devprime.agents.dqn import DQNAgent
from devprime.agents.ppo import PPOAgent
from devprime.utils.logger import Logger
from devprime.utils.visualizer import Visualizer
from devprime.utils.replay_buffer import ReplayBuffer

def train_dqn(env, episodes=200, batch_size=32):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = DQNAgent(obs_dim, act_dim)
    buffer = ReplayBuffer(10000, obs_dim)
    logger = Logger()
    rewards = []
    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.act(obs, epsilon=max(0.1, 1.0 - ep/100))
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            if buffer.cur_size > batch_size:
                batch = buffer.sample(batch_size)
                agent.update(batch)
        logger.log({'episode': ep, 'reward': ep_reward})
        rewards.append(ep_reward)
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}, Reward: {ep_reward:.2f}")
    agent.save('dqn_gridworld.pth')
    Visualizer().plot_rewards(rewards, label='DQN GridWorld')
    return rewards

def train_ppo(env, episodes=200, steps_per_update=128):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = PPOAgent(obs_dim, act_dim)
    logger = Logger()
    rewards = []
    obs = env.reset()
    ep_reward = 0
    batch_obs, batch_act, batch_logp, batch_rew, batch_done = [], [], [], [], []
    for ep in range(episodes):
        for t in range(steps_per_update):
            action, logp = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            batch_obs.append(obs)
            batch_act.append(action)
            batch_logp.append(logp)
            batch_rew.append(reward)
            batch_done.append(done)
            obs = next_obs
            ep_reward += reward
            if done:
                obs = env.reset()
                rewards.append(ep_reward)
                logger.log({'episode': ep, 'reward': ep_reward})
                ep_reward = 0
     
        returns = []
        G = 0
        for r, d in zip(reversed(batch_rew), reversed(batch_done)):
            G = r + agent.gamma * G * (1 - d)
            returns.insert(0, G)
        obs_tensor = torch.FloatTensor(batch_obs)
        values = agent.value(obs_tensor).detach().numpy()
        advs = np.array(returns) - values
        batch = (batch_obs, batch_act, batch_logp, returns, advs)
        agent.update(batch)
        batch_obs, batch_act, batch_logp, batch_rew, batch_done = [], [], [], [], []
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}, Reward: {rewards[-1]:.2f}")
    agent.save('ppo_gridworld.pth')
    Visualizer().plot_rewards(rewards, label='PPO GridWorld')
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gridworld', choices=['gridworld', 'robot'])
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ppo'])
    args = parser.parse_args()
    if args.env == 'gridworld':
        env = GridWorldEnv(size=5)
    else:
        env = RobotEnv(render=False)
    if args.algo == 'dqn':
        train_dqn(env)
    else:
        train_ppo(env)



import argparse

def train_dqn(env):
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    replay_buffer = ReplayBuffer(capacity=10000)
    rewards = []
    num_episodes = 200
    batch_size = 64
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)
        rewards.append(ep_reward)
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}, Reward: {ep_reward:.2f}")
    agent.save('dqn_gridworld.pth')
    Visualizer().plot_rewards(rewards, label='DQN GridWorld')
    return rewards

def train_ppo(env):
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
    rewards = []
    num_episodes = 200
    batch_obs, batch_act, batch_logp, batch_rew, batch_done = [], [], [], [], []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, logp = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            batch_obs.append(obs)
            batch_act.append(action)
            batch_logp.append(logp)
            batch_rew.append(reward)
            batch_done.append(done)
            obs = next_obs
            ep_reward += reward
        rewards.append(ep_reward)
        agent.update((batch_obs, batch_act, batch_logp, batch_rew, batch_done))
        batch_obs, batch_act, batch_logp, batch_rew, batch_done = [], [], [], [], []
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}, Reward: {ep_reward:.2f}")
    agent.save('ppo_gridworld.pth')
    Visualizer().plot_rewards(rewards, label='PPO GridWorld')
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gridworld', choices=['gridworld', 'robot'])
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ppo'])
    args = parser.parse_args()
    if args.env == 'gridworld':
        env = GridWorldEnv(size=5)
    else:
        env = RobotEnv(render=False)
    if args.algo == 'dqn':
        train_dqn(env)
    else:
        train_ppo(env)
