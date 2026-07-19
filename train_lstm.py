import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, NormalizeObservation, NormalizeReward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import DrawLine

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--model-name', type=str, help='model name used to save checkpoint.')
parser.add_argument('--model', type=str, default='net', help='net or net2')
parser.add_argument('--max-training-steps', type=int, default=10000, metavar='N', help='maximum number of training steps. incremented by 1 * ppo_epochs once buffer full (default: 10000)')
parser.add_argument('--max-episode-steps', type=int, default=1500, metavar='N', help='maximum number of steps in an episode (default: 1500)')
parser.add_argument('--ppo-epochs', type=int, default=5, metavar='N', help='(default: 5)')
parser.add_argument('--unroll-steps', type=int, default=128, metavar='N', help='number of steps per rollout (default: 128)')
parser.add_argument('--params-path', type=str, default=None, help='path to the saved model parameters')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--lambda_', type=float, default=0, metavar='G', help='GAE lambda factor (default: 0)')
parser.add_argument('--vf_coef', type=float, default=0.5, metavar='G', help='"coefficient of the value function in loss function" (default: 0.5)')
parser.add_argument('--lr', type=float, default=2.5e-4, metavar='G', help='learning rate of agent (default: 2.5e-4)')
parser.add_argument('--anneal-lr', action='store_true', default=False, help='Toggle learning rate annealing for policy and value networks')
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
parser.add_argument('--no-rewards-threshold', type=float, default=-0.1, metavar='G', help='threshold of average reward before terminating episode (default: -0.1)')
parser.add_argument('--num-lstm-layers', type=int, default=1, metavar='N', help='number of LSTM layers (default: 1)')
parser.add_argument('--action-repeat', type=int, default=4, metavar='N', help='repeat action in N frames (default: 4)')
parser.add_argument('--seed', type=int, default=125, metavar='N', help='random seed (default: 125)')
parser.add_argument('--scale-rewards', action='store_true', default=False, help='scale reward through NormalizeReward wrapper')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='interval between training status logs (default: 5)')
parser.add_argument('--load-weights', action='store_true', help='load pre-trained weights')
parser.add_argument("--target-kl", type=float, default=0.02, help="the target KL divergence threshold")
args = parser.parse_args()

run_name = f"{args.model_name}_{datetime.now().strftime('%b%d_%H%M')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    # These would absolutely make the training deterministic, but they would also slow down the training. Useful for debugging/tuning but not for actual training.
    torch.backends.cudnn.deterministic = True   # force cuDNN to pick a deterministic algorithm
    torch.backends.cudnn.benchmark = False       # prevents the auto-tuner from overriding that choice and selecting a nondeterministic algorithm that would be faster
    
transition = np.dtype([('o', np.float64, (1, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('next_o', np.float64, (1, 96, 96)), ('v', np.float64), ('next_die', np.int32), ('next_done', np.int32)])

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = NormalizeObservation(GrayscaleObservation(gym.make('CarRacing-v3', continuous=True, max_episode_steps=args.max_episode_steps), keep_dim=True))
        if args.scale_rewards:
            self.env = NormalizeReward(self.env)
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
            self.net = Net().double().to(device)
        elif args.model.lower() == 'net2':
            self.net = Net2().double().to(device)
        else:
            raise Exception('invalid model. choose net or net2.')
        self.buffer = np.empty(args.unroll_steps, dtype=transition)
        self.counter = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr) # starting lr was 1e-3

    def save_param(self, update_steps, score, running_score):
        checkpoint = {
                'model_state_dict': self.net.state_dict(),
                'update_steps': update_steps,
                'best_score': float(score),
                'running_score': float(running_score),
                'seed': float(args.seed)
            }
        torch.save(checkpoint, f'checkpoints/{args.model_name}.pkl')

    def load_param(self, path=None):
        if path is None:
            raise Exception('No model path provided.')
        checkpoint = torch.load(path, map_location=device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded update_steps {checkpoint['update_steps']}      Best_score {checkpoint['best_score']:.2f}      Best_running_score {checkpoint['running_score']:.2f}')
        return checkpoint['update_steps'], checkpoint['best_score'], checkpoint['running_score'], checkpoint['seed']

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == args.unroll_steps:
            self.counter = 0
            return True
        else:
            return False
    
    def select_action_and_value(self, state, lstm_state, next_done):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0) # converts shape to (seq_len=1, channels=1, 96, 96)
        with torch.no_grad():
            (alpha, beta, lstm_state), value = self.net(state, lstm_state, next_done)
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp, lstm_state, value
    
    def update(self, lstm_state, last_lstm_state_for_bootstrap):
        o = torch.tensor(self.buffer['o'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        next_o = torch.tensor(self.buffer['next_o'], dtype=torch.double).to(device)
        v = torch.tensor(self.buffer['v'], dtype=torch.double).to(device)
        v = v.view(-1, 1)
        next_die = torch.tensor(self.buffer['next_die'], dtype=torch.int32).to(device).view(-1, 1)
        next_done = torch.tensor(self.buffer['next_done'], dtype=torch.int32).to(device).view(-1, 1)
        next_terminal = 1 - (1 - next_done) * (1 - next_die)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)
        adv = torch.zeros(r.shape, dtype=torch.double).to(device)
        T=len(r)
        with torch.no_grad():
            v_last = self.net(next_o[-1].unsqueeze(0), last_lstm_state_for_bootstrap, next_die[-1])[1] # bootstrap
            v_next = torch.cat([v, v_last], 0)[1:]
            target_v = r + args.gamma * v_next * (1 - next_die) # if die (not trunc), penalize next state with just r.
            delta = target_v - v
        
        gae = 0
        for t in reversed(range(T)):
            gae = delta[t] + args.gamma * args.lambda_ * gae * (1 - next_die[t]) * (1 - next_done[t]) # if a state is die or done, then reset gae = delta[t] + 0
            adv[t] = gae

        mean_total_loss, mean_action_loss, mean_value_loss, mean_approx_kl = [], [], [], []

        # if used 's' and original 'terminal', in get_states(), it will zero-out LSTM hidden values
        # when predicting for 's' which still belongs to current episode. we only want to zero-out when starting next episode.
        terminal = torch.zeros_like(next_terminal)
        terminal[1:] = next_terminal[:-1]
        terminal[0] = 0 
        
        for i in range(args.ppo_epochs):
            index = range(args.unroll_steps)
            (alpha, beta, _), new_value = self.net(o[index], lstm_state, terminal[index])
            dist = Beta(alpha, beta)
            a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
            log_ratio = a_logp - old_a_logp[index]
            ratio = torch.exp(log_ratio)
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-log_ratio).mean()
                approx_kl = ((ratio - 1) - log_ratio).mean()
            if i == 0 and all(torch.round(ratio) != 1): print(round(ratio.item(), 5))
            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(new_value, target_v[index])
            loss = action_loss + args.vf_coef * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.training_step += 1
            mean_total_loss.append(loss.item())
            mean_action_loss.append(action_loss.item())
            mean_value_loss.append(value_loss.item())
            mean_approx_kl.append(approx_kl.item())
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    print(approx_kl)
        with torch.no_grad():
            (_, _, latest_lstm_state), _ = self.net(o[index], lstm_state, terminal[index])
        writer.add_scalar("Loss/total", np.mean(mean_total_loss), self.training_step)
        writer.add_scalar("Loss/action", np.mean(mean_action_loss), self.training_step)
        writer.add_scalar("Loss/value", np.mean(mean_value_loss), self.training_step)
        writer.add_scalar("KL/approx_kl", np.mean(mean_approx_kl), self.training_step)
        return latest_lstm_state

if __name__ == "__main__":
    agent = Agent()
    env = Env()
    running_score = 0
    best_score = float('-inf')
    best_running_score = float('-inf')
    if args.load_weights:
        agent.training_step, best_score, loaded_running_score, loaded_seed = agent.load_param(args.params_path)
        if loaded_seed != args.seed: 
            print (f'Loaded seed {loaded_seed} is different from input seed {args.seed}. Restart running score.')
        else:
            best_running_score = loaded_running_score
            running_score = loaded_running_score
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    training_records = []
    next_lstm_state = (
            torch.zeros(agent.net.lstm.num_layers, 1, agent.net.lstm.hidden_size, dtype=torch.double).to(device),
            torch.zeros(agent.net.lstm.num_layers, 1, agent.net.lstm.hidden_size, dtype=torch.double).to(device),
        )  # hidden and context states
    episode_num = 0
    while agent.training_step < args.max_training_steps:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (agent.training_step - 1.0) / args.max_training_steps
            lrnow = frac * args.lr
            if lrnow <= 0.1 * args.lr:
                lrnow = 0.1 * args.lr
            agent.optimizer.param_groups[0]["lr"] = lrnow
        score = 0
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        next_obs = env.reset() # state.shape (channels since keep_dim=True --> 96, 96, 1)
        next_obs = next_obs.reshape(1, 96, 96)
        next_done = torch.zeros(1).to(device)
        for t in range(args.max_episode_steps):
            # next_done and next_die tells you a new episode is starting after this timestep.
            action, a_logp, next_lstm_state, value = agent.select_action_and_value(next_obs, next_lstm_state, next_done)
            next_next_obs, reward, next_done, next_die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            next_next_obs = next_next_obs.reshape(1, 96, 96)
            if args.render:
                env.render()
            if agent.store((next_obs, action, a_logp, reward, next_next_obs, value, next_die, next_done)):
                next_lstm_state = agent.update(initial_lstm_state, next_lstm_state)
                initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
            score += reward
            next_obs = next_next_obs
            next_done = torch.tensor([next_done], dtype=torch.int32).to(device)
            if next_done or next_die:
                episode_num += 1
                next_lstm_state = (
                        torch.zeros(agent.net.lstm.num_layers, 1, agent.net.lstm.hidden_size, dtype=torch.double).to(device),
                        torch.zeros(agent.net.lstm.num_layers, 1, agent.net.lstm.hidden_size, dtype=torch.double).to(device),
                    )  # hidden and context states
                break
        running_score = running_score * 0.99 + score * 0.01
        if score > best_score:
            best_score = score

        if running_score > best_running_score:
            best_running_score = running_score
            agent.save_param(agent.training_step, score, best_running_score)
            print(f"New best running score: {best_running_score:.2f}, saved model parameters.\nBest score so far: {best_score:.2f}")

        if episode_num % args.log_interval == 0:
            if args.vis:
                draw_reward(xdata=agent.training_step, ydata=running_score)
            print('Episode {}\tUpd Step {}\tLast score: {:.2f}\tMoving average score: {:.2f}\tLearning rate: {:.6f}'.format(episode_num,agent.training_step, score, running_score, agent.optimizer.param_groups[0]["lr"]))
            writer.add_scalar("Score/raw", score, agent.training_step)
            writer.add_scalar("Score/running_avg", running_score, agent.training_step)

        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break

    writer.add_hparams(vars(args), {"best_score": best_score, "best_running": best_running_score})
    writer.close()