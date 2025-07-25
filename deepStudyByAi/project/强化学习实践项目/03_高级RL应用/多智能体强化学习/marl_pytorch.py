import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Define a simple Cooperative Grid World Environment
class CooperativeGridWorld:
    def __init__(self, size=5, num_agents=2):
        self.size = size
        self.num_agents = num_agents
        self.agents_pos = [(0, 0), (size - 1, 0)] # Initial positions for agents
        self.goals_pos = [(size - 1, size - 1), (0, size - 1)] # Goals for agents
        self.actions = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.action_map = {v: k for k, v in self.actions.items()}

    def reset(self):
        self.agents_pos = [(0, 0), (self.size - 1, 0)]
        return [self.agent_state_to_obs(i) for i in range(self.num_agents)]

    def agent_state_to_obs(self, agent_idx):
        # Observation for each agent: its own position and the positions of all goals
        obs = list(self.agents_pos[agent_idx])
        for goal in self.goals_pos:
            obs.extend(goal)
        return np.array(obs, dtype=np.float32)

    def step(self, actions_taken): # actions_taken is a list of actions for each agent
        rewards = [0] * self.num_agents
        dones = [False] * self.num_agents

        new_agents_pos = []
        for i, action_idx in enumerate(actions_taken):
            x, y = self.agents_pos[i]
            new_x, new_y = x, y

            action = self.actions[action_idx]

            if action == 'up':
                new_x = max(0, x - 1)
            elif action == 'down':
                new_x = min(self.size - 1, x + 1)
            elif action == 'left':
                new_y = max(0, y - 1)
            elif action == 'right':
                new_y = min(self.size - 1, y + 1)
            
            new_agents_pos.append((new_x, new_y))
        
        self.agents_pos = new_agents_pos # Update all agents' positions simultaneously

        # Check if all goals are reached
        all_goals_reached = True
        for i in range(self.num_agents):
            if self.agents_pos[i] != self.goals_pos[i]:
                all_goals_reached = False
                break

        if all_goals_reached:
            global_reward = 10.0 # Large positive reward for cooperative success
            done_all = True
        else:
            global_reward = -0.1 # Small penalty per step
            done_all = False

        # All agents receive the same global reward
        for i in range(self.num_agents):
            rewards[i] = global_reward
            dones[i] = done_all

        next_states = [self.agent_state_to_obs(i) for i in range(self.num_agents)]
        return next_states, rewards, dones, {}

    def render(self):
        grid = np.full((self.size, self.size), '.', dtype='U1')
        for i, pos in enumerate(self.agents_pos):
            grid[pos] = f'A{i}'
        for i, pos in enumerate(self.goals_pos):
            if grid[pos] == '.': # Don't overwrite agent if it's on a goal
                grid[pos] = f'G{i}'
        print("--------------------")
        for row in grid:
            print(" ".join(row))
        print("--------------------")

# Define a simple Q-Network for each agent
class AgentQNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=64):
        super(AgentQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, obs):
        x = self.fc1(obs)
        x = self.relu(x)
        q_values = self.fc2(x)
        return q_values

# Independent DQN Agent for MARL (for simplicity, using independent learners)
class IndependentDQNAgent:
    def __init__(self, obs_size, action_size, learning_rate=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, replay_buffer_capacity=5000, batch_size=32, target_update_freq=50):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.q_network = AgentQNetwork(obs_size, action_size)
        self.target_q_network = AgentQNetwork(obs_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.replay_buffer = deque(maxlen=replay_buffer_capacity)

    def choose_action(self, obs):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                q_values = self.q_network(obs_tensor)
                return torch.argmax(q_values).item()

    def store_experience(self, obs, action, reward, next_obs, done):
        self.replay_buffer.append((obs, action, reward, next_obs, done))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs_tensor = torch.FloatTensor(np.array(obs))
        actions_tensor = torch.LongTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(np.array(rewards))
        next_obs_tensor = torch.FloatTensor(np.array(next_obs))
        dones_tensor = torch.BoolTensor(np.array(dones))

        current_q_values = self.q_network(obs_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values_target = self.target_q_network(next_obs_tensor).max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values_target * (~dones_tensor)

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

# Training loop for MARL
if __name__ == "__main__":
    env = CooperativeGridWorld()
    obs_size = len(env.reset()[0]) # State for each agent
    action_size = len(env.actions)
    num_agents = env.num_agents

    agents = [IndependentDQNAgent(obs_size, action_size) for _ in range(num_agents)]

    num_episodes = 2000
    total_rewards_per_episode = []

    print("\n--- Training Multi-Agent (Independent DQN) on Cooperative Grid World ---")
    for episode in range(num_episodes):
        obs_n = env.reset() # obs_n is a list of observations, one for each agent
        done_n = [False] * num_agents
        episode_global_reward = 0
        step_count = 0

        while not all(done_n) and step_count < 200: # Max steps per episode
            actions_n = []
            for i, obs in enumerate(obs_n):
                action = agents[i].choose_action(obs)
                actions_n.append(action)

            next_obs_n, rewards_n, done_n, _ = env.step(actions_n)

            # For independent learners, each agent stores its own experience and learns from it.
            # In this cooperative setup, all agents receive the same global reward.
            for i in range(num_agents):
                agents[i].store_experience(obs_n[i], actions_n[i], rewards_n[i], next_obs_n[i], done_n[i])
                agents[i].learn()

            obs_n = next_obs_n
            episode_global_reward += rewards_n[0] # All agents get same reward
            step_count += 1
        
        # Epsilon decay for all agents
        for agent in agents:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        total_rewards_per_episode.append(episode_global_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(total_rewards_per_episode[-50:])
            print(f"Episode {episode + 1}: Avg Global Reward = {avg_reward:.2f}, Epsilon = {agents[0].epsilon:.2f}")

    print("\n--- Training Finished ---")

    # Test the trained agents
    print("\n--- Testing Learned Multi-Agent Policy ---")
    test_env = CooperativeGridWorld(size=5, num_agents=2)
    obs_n = test_env.reset()
    done_n = [False] * num_agents
    test_env.render()
    test_steps = 0
    while not all(done_n) and test_steps < 50: # Limit test steps
        actions_n = []
        for i, obs in enumerate(obs_n):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                q_values = agents[i].q_network(obs_tensor)
                action = torch.argmax(q_values).item() # Choose deterministic action for testing
            actions_n.append(action)
        
        next_obs_n, rewards_n, done_n, _ = test_env.step(actions_n)
        obs_n = next_obs_n
        test_env.render()
        test_steps += 1
    print(f"All goals reached: {all(done_n)}")
    print(f"Test completed in {test_steps} steps.") 