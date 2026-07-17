import argparse

import numpy as np

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, NormalizeObservation
#from train import Net
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--params-path', type=str, default=None, help='path to the saved model parameters')
parser.add_argument('--max-episode-steps', type=int, default=1000, metavar='N', help='maximum number of steps in an episode (default: 1000)')
parser.add_argument('--num-episodes', type=int, default=5, metavar='N', help='number of episodes to run (default: 5)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


class Env():
    """
    Test environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = NormalizeObservation(GrayscaleObservation(gym.make('CarRacing-v3', render_mode='human', continuous=True, max_episode_steps=args.max_episode_steps)))
        spec = gym.spec('CarRacing-v3')
        self.reward_threshold = spec.reward_threshold if spec.reward_threshold else float('inf')

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        observation, _ = self.env.reset(seed=args.seed)
        self.stack = [np.array(observation)] * args.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            observation, reward, die, trunc, _ = self.env.step(action)
            reward = float(reward)
            # don't penalize "die state"
            if die:
                reward += 100
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 or trunc else False
            if done or die:
                break
        self.stack.pop(0)
        self.stack.append(np.array(observation))
        assert len(self.stack) == args.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v
    
class Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = Net().double().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta) # mean of beta distribution for greedy action.

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self, path=None):
        if path is None:
            path = 'param/ppo_net_params.pkl'
        checkpoint = torch.load(path, map_location=device)
        self.net.load_state_dict(checkpoint['model_state_dict'])


if __name__ == "__main__":
    agent = Agent()
    agent.load_param(args.params_path)
    env = Env()

    running_score = 0
    state = env.reset()
    for i_ep in range(args.num_episodes):
        score = 0
        state = env.reset()

        for t in range(args.max_episode_steps):
            action = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
    env.env.close()
