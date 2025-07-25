import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gym

# Define the Q-Network for Game AI
class GameQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(GameQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# Game AI Agent (DQN-based)
class GameAIAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, replay_buffer_capacity=10000, batch_size=64, target_update_freq=100):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.q_network = GameQNetwork(state_size, action_size)
        self.target_q_network = GameQNetwork(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.BoolTensor(dones)

        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values_target = self.target_q_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values_target * (~dones_tensor)

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

# Training loop for Game AI
if __name__ == "__main__":
    env = gym.make('CartPole-v1') # Using CartPole as a simple game environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = GameAIAgent(state_size, action_size)

    num_episodes = 500
    scores = []

    print("\n--- Training Game AI (DQN on CartPole) ---")
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            agent.learn()

            state = next_state
            episode_reward += reward

        scores.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"Episode {episode + 1}: Average Score = {avg_score:.2f}, Epsilon = {agent.epsilon:.2f}")

    env.close()
    print("\n--- Training Finished ---")

    # Test the trained Game AI agent
    print("\n--- Testing Learned Game AI Policy ---")
    test_env = gym.make('CartPole-v1', render_mode='human') # Render for visualization
    state = test_env.reset()[0]
    done = False
    test_reward = 0
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = agent.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        
        next_state, reward, done, _, _ = test_env.step(action)
        state = next_state
        test_reward += reward
    test_env.close()
    print(f"Test Episode Total Reward: {test_reward:.2f}") 