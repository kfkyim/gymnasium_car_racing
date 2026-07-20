import argparse
from datetime import datetime
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, NormalizeObservation, NormalizeReward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from utils import DrawLine

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--model', type=str, default='net', help='net or net2')
parser.add_argument('--num-episodes', type=int, default=3, metavar='N', help='number of episodes to test (default: 3)')
parser.add_argument('--max-episode-steps', type=int, default=1500, metavar='N', help='maximum number of steps in an episode (default: 1500)')
parser.add_argument('--params-path', type=str, default=None, help='path to the saved model parameters')
parser.add_argument('--no-rewards-threshold', type=float, default=-0.1, metavar='G', help='threshold of average reward before terminating episode (default: -0.1)')
parser.add_argument('--num-lstm-layers', type=int, default=1, metavar='N', help='number of LSTM layers (default: 1)')
parser.add_argument('--action-repeat', type=int, default=4, metavar='N', help='repeat action in N frames (default: 4)')
parser.add_argument('--seed', type=int, default=125, metavar='N', help='random seed (default: 125)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    # These would absolutely make the training deterministic, but they would also slow down the training. Useful for debugging/tuning but not for actual training.
    torch.backends.cudnn.deterministic = True   # force cuDNN to pick a deterministic algorithm
    torch.backends.cudnn.benchmark = False       # prevents the auto-tuner from overriding that choice and selecting a nondeterministic algorithm that would be faster
    
class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = NormalizeObservation(GrayscaleObservation(gym.make('CarRacing-v3', continuous=True, max_episode_steps=args.max_episode_steps, render_mode='human'), keep_dim=True))
        spec = gym.spec('CarRacing-v3')
        self.reward_threshold = spec.reward_threshold if spec.reward_threshold else float('inf')
        self.max_episode_steps = spec.max_episode_steps if spec.max_episode_steps else float('inf')

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        observation, _ = self.env.reset(seed=args.seed)
        return np.array(observation)

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            observation, reward, die, trunc, _ = self.env.step(action)
            reward = float(reward)
            # don't penalize "die state"
            # if die:
            #     reward += 100

            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= args.no_rewards_threshold or trunc else False
            if done or die:
                break
        return np.array(observation), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
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
        self.cnn_base = nn.Sequential(  # input shape (# channels, 96, 96)
            nn.Conv2d(1, 8, kernel_size=4, stride=2),
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
        self.cnn_fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU()) # output shape: (1, 128)
        self.lstm = nn.LSTM(128, 64, args.num_lstm_layers)
        self.v = nn.Linear(64, 1)
        self.alpha_head = nn.Sequential(nn.Linear(64, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(64, 3), nn.Softplus())

        self.cnn_base.apply(self._cnn_weights_init)
        self.cnn_fc.apply(self._cnn_weights_init)
        
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.v.apply(self._value_weights_init)
        self.alpha_head.apply(self._policy_weights_init)
        self.beta_head.apply(self._policy_weights_init)

    @staticmethod
    def _cnn_weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _value_weights_init(m):
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0) 

    @staticmethod
    def _policy_weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0) 

    def _get_states(self, x, lstm_state, done):
         # x.shape = (seq_len, channel=1, 96, 96)
        x = self.cnn_base(x) # (1, 256, 1, 1)
        x = x.reshape(x.shape[0], 1, 1, 256) # (1, 1, 1, 256)
        x = self.cnn_fc(x) # x.shape = (1, 1, 1, 128)
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        done = done.reshape((-1, batch_size)) # change from shape (1) to (1, 1)
        new_hidden = []
        for i, d in zip(x, done):
            # nn.LSTM( input, (h_t-1, c_t-1) ) -> output, (h_t, c_t), where h = hidden and c = context/cell states
            i, lstm_state = self.lstm(
                i, # i.shape [1, 1, 128]
                (
                    (1.0 - d) * lstm_state[0],
                    (1.0 - d) * lstm_state[1],
                ),
            )
            new_hidden += [i]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    
    def forward(self, x, lstm_state, done):
        hidden, lstm_state = self._get_states(x, lstm_state, done)
        v = self.v(hidden)
        alpha = self.alpha_head(hidden) + 1
        beta = self.beta_head(hidden) + 1
        return (alpha, beta, lstm_state), v

class Net2(Net):
    def __init__(self):
        super(Net2, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (# channels, 96, 96)
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),  # activation
            nn.Flatten()
        ) 
        self.cnn_fc = nn.Sequential(nn.Linear(4096, 512), nn.ReLU()) # output shape: (1, 128)
        self.lstm = nn.LSTM(512, 128, args.num_lstm_layers)
        self.v = nn.Linear(128, 1)
        self.alpha_head = nn.Sequential(nn.Linear(128, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(128, 3), nn.Softplus())

        self.cnn_base.apply(self._cnn_weights_init)
        self.cnn_fc.apply(self._cnn_weights_init)
        
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.v.apply(self._value_weights_init)
        self.alpha_head.apply(self._policy_weights_init)
        self.beta_head.apply(self._policy_weights_init)

    def _get_states(self, x, lstm_state, done):
         # x.shape = (seq_len, channel=1, 96, 96)
        x = self.cnn_base(x) # (1, 4096, 1, 1)
        x = x.reshape(x.shape[0], 1, 1, 4096)
        x = self.cnn_fc(x) # x.shape = (1, 1, 1, 512)
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        done = done.reshape((-1, batch_size)) # change from shape (1) to (1, 1)
        new_hidden = []
        for i, d in zip(x, done):
            # nn.LSTM( input, (h_t-1, c_t-1) ) -> output, (h_t, c_t), where h = hidden and c = context/cell states
            i, lstm_state = self.lstm(
                i, # i.shape [1, 1, 512]
                (
                    (1.0 - d) * lstm_state[0],
                    (1.0 - d) * lstm_state[1],
                ),
            )
            new_hidden += [i]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    def __init__(self):
        self.training_step = 0
        if args.model.lower() == 'net':
            self.net = Net().float().to(device)
        elif args.model.lower() == 'net2':
            self.net = Net2().float().to(device)
        else:
            raise Exception('invalid model. choose net or net2.')

    def load_param(self, path=None):
        if path is None:
            raise Exception('No model path provided.')
        checkpoint = torch.load(path, map_location=device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded update_steps {checkpoint['update_steps']}      Best_score {checkpoint['best_score']:.2f}      Best_running_score {checkpoint['running_score']:.2f}')
        return checkpoint['update_steps'], checkpoint['episode'], checkpoint['best_score'], checkpoint['running_score'], checkpoint['seed']
    
    def select_action_and_value(self, state, lstm_state, next_done):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0) # converts shape to (seq_len=1, channels=1, 96, 96)
        with torch.no_grad():
            (alpha, beta, lstm_state), value = self.net(state, lstm_state, next_done)
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp, lstm_state, value

if __name__ == "__main__":
    agent = Agent()
    env = Env()
    episode_num = 0
    agent.training_step, loaded_episode, best_score, loaded_running_score, loaded_seed = agent.load_param(args.params_path)
    training_records = []
    
    for i_ep in range(0, args.num_episodes):
        score = 0
        next_lstm_state = (
            torch.zeros(agent.net.lstm.num_layers, 1, agent.net.lstm.hidden_size, dtype=torch.float32).to(device),
            torch.zeros(agent.net.lstm.num_layers, 1, agent.net.lstm.hidden_size, dtype=torch.float32).to(device),
        )  # hidden and context states
        next_obs = env.reset() # state.shape (channels since keep_dim=True --> 96, 96, 1)
        next_obs = next_obs.reshape(1, 96, 96)
        next_done = torch.zeros(1).to(device)
        for t in range(args.max_episode_steps):
            # next_done and next_die tells you a new episode is starting after this timestep.
            action, a_logp, next_lstm_state, value = agent.select_action_and_value(next_obs, next_lstm_state, next_done)
            next_next_obs, reward, next_done, next_die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            next_next_obs = next_next_obs.reshape(1, 96, 96)
            score += reward
            next_obs = next_next_obs
            next_done = torch.tensor([next_done], dtype=torch.int32).to(device)
            if next_done or next_die:
                break
        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
       