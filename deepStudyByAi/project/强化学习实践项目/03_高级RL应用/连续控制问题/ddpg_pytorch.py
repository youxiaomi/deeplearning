import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym # Using gymnasium
import torch.nn.functional as F # Added missing import

# Define Actor Network (Policy Network)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Scale output to action space range using tanh and max_action
        return self.max_action * torch.tanh(self.fc3(x))

# Define Critic Network (Q-Value Network)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Concatenate state and action for critic input
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005, replay_buffer_capacity=int(1e6), batch_size=256):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.action_dim = action_dim

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state_tensor).cpu().data.numpy().flatten()

    def train(self):
        if len(self.replay_buffer) < self.batch_size: # Not enough experiences to train
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Compute target Q-value
        with torch.no_grad():
            next_action = self.actor_target(next_state) # Deterministic action from target actor
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q

        # Get current Q-value
        current_q = self.critic(state, action)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        # We want to maximize Q(s, actor(s)), so minimize -Q(s, actor(s))
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# Ornstein-Uhlenbeck Noise for exploration
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state

# Training loop
if __name__ == "__main__":
    env = gym.make('Pendulum-v1', render_mode=None) # Set render_mode=None for faster training
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)
    noise = OUNoise(action_dim)

    num_episodes = 200 # Pendulum usually converges quickly
    episode_rewards = []

    print("\n--- Training DDPG on Pendulum-v1 ---")
    for episode in range(num_episodes):
        state, _ = env.reset()
        noise.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done and step_count < 200: # Max steps per episode for Pendulum
            action = agent.select_action(state)
            action = (action + noise.noise()).clip(-max_action, max_action) # Add noise for exploration

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            episode_reward += reward
            step_count += 1

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}: Average Reward = {avg_reward:.2f}")

    env.close()
    print("\n--- Training Finished ---")

    # Test the trained agent
    print("\n--- Testing Learned DDPG Policy ---")
    test_env = gym.make('Pendulum-v1', render_mode='human') # Render for visualization
    state, _ = test_env.reset()
    done = False
    test_reward = 0
    test_steps = 0
    while not done and test_steps < 200:
        action = agent.select_action(state) # No noise during testing
        next_state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        state = next_state
        test_reward += reward
        test_steps += 1
    test_env.close()
    print(f"Test Episode Total Reward: {test_reward:.2f}") 