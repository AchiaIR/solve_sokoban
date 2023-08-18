
import copy
import imageio
import numpy as np
from tqdm import tqdm
from torch.optim import SGD
from dnn_models.basic_model import *
from rl_utils.replay_memory import *


BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN():

    def __init__(self, env, epsilon, decay_rate, discount_factor, change_reward):
        self.original_env = env
        self.epsilon = epsilon
        self.decay = decay_rate
        self.discount_factor = discount_factor
        self.change_reward = change_reward
        self.policy = QNN().to(device)
        self.target = QNN().to(device)
        self.buffer = ReplayMemory(REPLAY_MEMORY_SIZE, BATCH_SIZE)
        self.optimizer = SGD(self.policy.parameters(), lr=0.0001)
        self.update_target(self.target)

    def update_target(self, target):
        target.load_state_dict(self.policy.state_dict())

    def get_state_tensor(self, state, done=False):
        if done:
            return None
        return torch.tensor(state, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    def update_epsilon(self):
        self.epsilon *= self.decay
        if self.epsilon < 0.05:
            self.epsilon = 0.05

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy(state).max(1)[1].view(1, 1)

    def calculate_reward(self):
        reward = 0
        if self.change_reward:
            room_state = self.env.room_state
            room_fixed = self.env.room_fixed
            box_x, box_y= np.where(room_state == 4)
            target_x, target_y = np.where(room_fixed == 2)
            for i in range(len(box_x)):
                for j in range(len(target_x)):
                    # Calculate distance between a box and a target
                    reward -= abs(box_x[i] - target_x[j]) + abs(box_y[i] - target_y[j])
        return float(reward)

    def update_model(self):
        "The heart of DQN Algorithm"
        batch = self.buffer.sample()
        state_batch = torch.cat(batch.s)
        action_batch = torch.cat(batch.a)
        reward_batch = torch.cat(batch.r)
        next_state_batch = torch.cat([ns for ns in batch.ns if ns is not None])
        non_final_mask = torch.tensor(tuple(map(lambda ns: ns is not None, batch.ns)), device=device, dtype=torch.bool)

        # Compute Q-values for the current state and selected actions
        q_values = self.policy(state_batch).gather(1, action_batch)

        # Compute the maximum Q-values for the next states
        # use target network to estimate these Q-values to avoid biasing
        # the estimates with the values produced by the policy network
        next_q_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_q_values[non_final_mask] = self.target(next_state_batch).max(1)[0]

        # Compute the target Q-values
        target_q_values = (next_q_values * self.discount_factor) + reward_batch

        # Compute MSE Loss between predicted and target Q-values
        criterion = nn.MSELoss()
        loss = criterion(q_values, target_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes):
        losses, rewards = [], []
        for i in tqdm(range(num_episodes)):
            self.env = copy.deepcopy(self.original_env)
            state = self.get_state_tensor(self.env.get_image(mode='rgb_array'))
            done = False
            episode_reward = 0
            episode_loss = 0
            num_steps = 0
            while not done:
                action = self.epsilon_greedy_action(state)
                self.update_epsilon()
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = self.get_state_tensor(next_state, done)
                reward += self.calculate_reward()
                episode_reward += reward
                reward = torch.tensor([reward], device=device)
                self.buffer.push(state, action, next_state, reward)
                state = next_state
                if len(self.buffer) >= BATCH_SIZE:
                    loss = self.update_model()
                    episode_loss += loss
                    num_steps += 1

            rewards.append(episode_reward)
            losses.append(episode_loss / num_steps)

            # Once in 10 episodes - update target network
            if i % 10 == 9:
                self.update_target(self.target)

            # save the MidWay parameters to show
            if num_episodes / 2 == i:
                self.MidWay = QNN().to(device)
                self.update_target(self.MidWay)

        return losses, rewards

    def test(self, fname, MidWay=False):
        env = copy.deepcopy(self.original_env)
        done = False
        iter = 0
        state = env.get_image(mode='rgb_array')
        video_filename = fname + '.mp4'
        policy = self.policy if MidWay is False else self.MidWay
        with imageio.get_writer(video_filename, fps=10) as video:
            video.append_data(env.render(mode='rgb_array'))
            while (iter < 10) or not done:
                action = policy(self.get_state_tensor(state)).max(1)[1].view(1, 1)
                act = np.array(action.cpu())[0][0]
                state, reward, done, info = env.step(act)
                iter += 1
                video.append_data(env.render(mode='rgb_array'))
        return video_filename