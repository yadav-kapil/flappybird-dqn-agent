import argparse
import itertools
import json
import os
import random
import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dqn import DQN
from experience_replay import ReplayMemory

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "runs"), exist_ok=True)


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        yaml_path = os.path.join(BASE_DIR, "parameters.yaml")
        with open(yaml_path, "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(BASE_DIR, "runs", f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(BASE_DIR, "runs", f"{self.param_set}.pt")
        self.META_FILE = os.path.join(BASE_DIR, "runs", f"{self.param_set}.json")

    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            if os.path.exists(self.MODEL_FILE) and os.path.exists(self.META_FILE):
                policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
                with open(self.META_FILE, "r") as f:
                    meta = json.load(f)
                    best_reward = meta.get("best_reward", float("-inf"))
                    epsilon = meta.get("epsilon", self.epsilon_init)
                    start_episode = meta.get("episode", 0)
                print(
                    f"Resuming training from {self.MODEL_FILE} "
                    f"(Episode: {start_episode}, Best Reward: {best_reward:.2f}, Epsilon: {epsilon:.4f})"
                )
            else:
                best_reward = float("-inf")
                epsilon = self.epsilon_init
                start_episode = 0

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()
            epsilon = 0.0
            start_episode = 0

        for episode in itertools.count(start_episode):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0.0
            terminated = False

            while not terminated and episode_reward < self.reward_threshold:
                if is_training and random.random() < epsilon:
                    action = 1 if random.random() < 0.2 else 0
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                terminated = terminated or truncated

                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state
                episode_reward += reward.item()

            print(f"episode={episode+1} with total reward={episode_reward:.2f} & epsilon={epsilon:.4f}")

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if episode_reward > best_reward:
                    log_msg = f"Best Reward : {episode_reward} for Episode : {episode+1}"
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                    checkpoint_meta = {
                        "best_reward": episode_reward,
                        "epsilon": epsilon,
                        "episode": episode + 1,
                    }
                    with open(self.META_FILE, "w") as f:
                        json.dump(checkpoint_meta, f, indent=4)

                    best_reward = episode_reward

            if is_training and len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate Deep Q-Learning agent on Flappy Bird.")
    parser.add_argument("hyperparameters", help="Name of configuration profile in parameters.yaml")
    parser.add_argument("--train", action="store_true", help="Enable training mode (default is evaluation mode)")
    args = parser.parse_args()

    agent = Agent(param_set=args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)
