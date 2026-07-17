import argparse

import numpy as np

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, NormalizeObservation 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import DrawLine

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--max-episode-steps', type=int, default=1000, metavar='N', help='maximum number of steps in an episode (default: 1000)')
parser.add_argument('--unroll-steps', type=int, default=128, metavar='N', help='number of steps per rollout (default: 128)')
parser.add_argument('--params-path', type=str, default=None, help='path to the saved model parameters')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--lambda_', type=float, default=0, metavar='G', help='GAE lambda factor (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='G', help='learning rate of agent (default: 1e-4)')
parser.add_argument('--action-repeat', type=int, default=4, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='interval between training status logs (default: 5)')
parser.add_argument('--load-weights', action='store_true', help='load pre-trained weights')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    # These would absolutely make the training deterministic, but they would also slow down the training. Useful for debugging/tuning but not for actual training.
    # torch.backends.cudnn.deterministic = True   # force cuDNN to pick a deterministic algorithm
    # torch.backends.cudnn.benchmark = False       # prevents the auto-tuner from overriding that choice and selecting a nondeterministic algorithm that would be faster
    
transition = np.dtype([('s', np.float64, (1, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (1, 96, 96)), ('die', np.int32), ('done', np.int32)])

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = NormalizeObservation(GrayscaleObservation(gym.make('CarRacing-v3', continuous=True, max_episode_steps=args.max_episode_steps), keep_dim=True))
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
            if die:
                reward += 100

            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 or trunc else False
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
        # input shape expected: (1, 256) --> use .view(-1, 256) for this in _get_states().
        self.cnn_fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU()) # output shape: (1, 128)
        self.lstm = nn.LSTM(128, 64)
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

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 5
    buffer_capacity = 2096
    def __init__(self):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4) # starting lr was 1e-3

    def save_param(self, episode_num, score, running_score):
        checkpoint = {
                'model_state_dict': self.net.state_dict(),
                'episode_num': episode_num,
                'best_score': float(score),
                'running_score': float(running_score),
                'seed': float(args.seed)
            }
        torch.save(checkpoint, f'lstm_max_ep_steps_{args.max_episode_steps}_action_repeat_{args.action_repeat}_lambda_{args.lambda_}_seed_{args.seed}.pkl')

    def load_param(self, path=None):
        if path is None:
            path = f'param/params_max_ep_steps_{args.max_episode_steps}_action_repeat_{args.action_repeat}_lambda_{args.lambda_}_seed_{args.seed}.pkl'
        checkpoint = torch.load(path, map_location=device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded episode_num {checkpoint['episode_num']}      Best_score {checkpoint['best_score']:.2f}      Best_running_score {checkpoint['running_score']:.2f}')
        return checkpoint['episode_num'], checkpoint['best_score'], checkpoint['running_score'], checkpoint['seed']

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False
    
    def select_action(self, state, lstm_state, done):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0) # converts shape to (seq_len=1, channels=1, 96, 96)
        with torch.no_grad():
            alpha, beta, lstm_state = self.net(state, lstm_state, done)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp, lstm_state
    
    def update(self, lstm_state):
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)
        die = torch.tensor(self.buffer['die'], dtype=torch.int32).to(device).view(-1, 1)
        done = torch.tensor(self.buffer['done'], dtype=torch.int32).to(device).view(-1, 1)
        terminal = 1 - (1 - done) * (1 - die)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)
        adv = torch.zeros(r.shape, dtype=torch.double).to(device)
        T=len(r)
        
        with torch.no_grad():
            v = self.net(s, lstm_state, terminal)[1] # s.shape -> (seq_len)
            v_next = self.net(s_, lstm_state, terminal)[1]
            not_die = 1 # (1 - die) normally, we would penalize the value of the next state if the current state is die, but the original author of this fork didn't penalize and it worked well. So we will keep it as 1.
            target_v = r + args.gamma * v_next * not_die
            delta = target_v - v
        
        gae = 0
        for t in reversed(range(T)):
            gae = delta[t] + args.gamma * args.lambda_ * gae * (1 - die[t]) * (1 - done[t]) # if a state is die or done, then reset gae = delta[t] + 0
            adv[t] = gae

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(range(self.buffer_capacity), args.unroll_steps, False):

                alpha, beta, _ = self.net(s[index], lstm_state, terminal[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])
                if self.training_step == 0: assert all(ratio == 1)
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index], lstm_state, terminal[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.training_step += 1


if __name__ == "__main__":
    agent = Agent()
    env = Env()
    episode_num = 0
    running_score = 0
    best_score = float('-inf')
    best_running_score = float('-inf')
    if args.load_weights:
        episode_num, best_score, loaded_running_score, loaded_seed = agent.load_param(args.params_path)
        if loaded_seed != args.seed: 
            print (f'Loaded seed {loaded_seed} is different from input seed {args.seed}. Restart running score.')
        else:
            best_running_score = loaded_running_score
            running_score = loaded_running_score
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    training_records = []
    
    for i_ep in range(episode_num, 100000):
        score = 0
        next_lstm_state = (
            torch.zeros(1, agent.net.lstm.num_layers, agent.net.lstm.hidden_size, dtype=torch.double).to(device),
            torch.zeros(1, agent.net.lstm.num_layers, agent.net.lstm.hidden_size, dtype=torch.double).to(device),
        )  # hidden and context states
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        state = env.reset() # state.shape (channels since keep_dim=True --> 96, 96, 1)
        state = state.reshape(1, 96, 96)
        next_done = torch.zeros(1).to(device)
        for t in range(args.max_episode_steps): # max number of steps in an episode
            action, a_logp, next_lstm_state = agent.select_action(state, next_lstm_state, next_done)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            state_ = state_.reshape(1, 96, 96)
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_, die, done)):
                print('updating')
                agent.update(initial_lstm_state)
            score += reward
            state = state_
            next_done = torch.tensor([done], dtype=torch.int32).to(device)
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01
        if score > best_score:
            best_score = score

        if running_score > best_running_score:
            best_running_score = running_score
            agent.save_param(i_ep, score, best_running_score)
            print(f"New best running score: {best_running_score:.2f}, saved model parameters.\nBest score so far: {best_score:.2f}")

        if i_ep % args.log_interval == 0:
            if args.vis:
                draw_reward(xdata=i_ep, ydata=running_score)
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))

        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
