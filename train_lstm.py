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
parser.add_argument('--params-path', type=str, default=None, help='path to the saved model parameters')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--lambda_', type=float, default=0, metavar='G', help='GAE lambda factor (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='G', help='learning rate of agent (default: 1e-4)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='interval between training status logs (default: 1)')
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
    
transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96)), ('die', np.int32), ('done', np.int32)])

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = NormalizeObservation(GrayscaleObservation(gym.make('CarRacing-v3', continuous=True, max_episode_steps=args.max_episode_steps)))
        spec = gym.spec('CarRacing-v3')
        self.reward_threshold = spec.reward_threshold if spec.reward_threshold else float('inf')
        self.max_episode_steps = spec.max_episode_steps if spec.max_episode_steps else float('inf')

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        observation, _ = self.env.reset(seed=args.seed)
        self.stack = [np.array(observation)] * args.img_stack  # four frames for decision
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
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 5
    buffer_capacity = 2000
    batch_size = 128
    def __init__(self):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4) # starting lr was 1e-3

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self, episode_num, score, running_score):
        checkpoint = {
                'model_state_dict': self.net.state_dict(),
                'episode_num': episode_num,
                'best_score': float(score),
                'running_score': float(running_score),
                'seed': float(args.seed)
            }
        torch.save(checkpoint, f'param/params_max_ep_steps_{args.max_episode_steps}_action_repeat_{args.action_repeat}_img_stack_{args.img_stack}_lambda_{args.lambda_}_seed_{args.seed}.pkl')

    def load_param(self, path=None):
        if path is None:
            path = f'param/params_max_ep_steps_{args.max_episode_steps}_action_repeat_{args.action_repeat}_img_stack_{args.img_stack}_lambda_{args.lambda_}_seed_{args.seed}.pkl'
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

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)
        die = torch.tensor(self.buffer['die'], dtype=torch.int32).to(device).view(-1, 1)
        done = torch.tensor(self.buffer['done'], dtype=torch.int32).to(device).view(-1, 1)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)
        adv = torch.zeros(r.shape, dtype=torch.double).to(device)
        T=len(r)
        with torch.no_grad():
            v = self.net(s)[1]
            v_next = self.net(s_)[1]
            not_die = 1 # (1 - die) normally, we would penalize the value of the next state if the current state is die, but the original author of this fork didn't penalize and it worked well. So we will keep it as 1.
            target_v = r + args.gamma * v_next * not_die
            delta = target_v - v
        
        gae = 0
        for t in reversed(range(T)):
            gae = delta[t] + args.gamma * args.lambda_ * gae * (1 - die[t]) * (1 - done[t]) # if a state is die or done, then reset gae = delta[t] + 0
            adv[t] = gae

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


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
    
    state = env.reset()
    for i_ep in range(episode_num, 100000):
        score = 0
        state = env.reset()

        for t in range(args.max_episode_steps): # max number of steps in an episode
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_, die, done)):
                print('updating')
                agent.update()
            score += reward
            state = state_
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
