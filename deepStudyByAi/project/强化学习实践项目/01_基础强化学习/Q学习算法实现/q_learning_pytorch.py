import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the Grid World environment
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start_state = (0, 0)
        self.goal_state = (size - 1, size - 1)
        self.obstacle_state = (size // 2, size // 2) # Example obstacle
        self.grid[self.goal_state] = 1 # Goal
        self.grid[self.obstacle_state] = -1 # Obstacle
        self.current_state = self.start_state
        self.actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action_idx):
        if action_idx not in self.action_map:
            raise ValueError("Invalid action")

        x, y = self.current_state
        new_x, new_y = x, y

        action = self.action_map[action_idx]

        if action == 'up':
            new_x = max(0, x - 1)
        elif action == 'down':
            new_x = min(self.size - 1, x + 1)
        elif action == 'left':
            new_y = max(0, y - 1)
        elif action == 'right':
            new_y = min(self.size - 1, y + 1)

        new_state = (new_x, new_y)
        reward = -0.1 # Small penalty for each step

        if new_state == self.goal_state:
            reward = 10.0
            done = True
        elif new_state == self.obstacle_state:
            reward = -5.0
            done = True
        else:
            done = False

        self.current_state = new_state
        return new_state, reward, done

    def render(self):
        display_grid = np.copy(self.grid).astype(str)
        display_grid[self.current_state] = 'A' # Agent
        display_grid[self.goal_state] = 'G' # Goal
        display_grid[self.obstacle_state] = 'O' # Obstacle
        print("--------------------")
        for row in display_grid:
            print(" ".join(row))
        print("--------------------")

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        q_values = self.fc2(x)
        return q_values

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size) # Explore
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item() # Exploit

    def learn(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        next_state_tensor = torch.FloatTensor(next_state)

        # Get current Q-value
        current_q_values = self.q_network(state_tensor)
        current_q = current_q_values.gather(0, action_tensor)

        # Get next Q-value
        next_q_values = self.q_network(next_state_tensor)
        max_next_q = torch.max(next_q_values)

        # Calculate target Q-value
        target_q = reward_tensor + (1 - done) * self.gamma * max_next_q

        # Compute loss and update Q-network
        loss = self.criterion(current_q, target_q.detach()) # detach target_q to avoid computing gradients for it
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Training loop
if __name__ == "__main__":
    env = GridWorld()
    state_size = 2 # (x, y) coordinates
    action_size = len(env.actions)
    agent = QLearningAgent(state_size, action_size)

    num_episodes = 500

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")

    print("\nTraining finished. Testing the learned policy:")
    state = env.reset()
    env.render()
    done = False
    test_steps = 0
    while not done and test_steps < 50: # Limit test steps to avoid infinite loops
        action = agent.choose_action(state) # Use learned policy (epsilon is small)
        next_state, reward, done = env.step(action)
        state = next_state
        env.render()
        test_steps += 1
    print(f"Reached goal in {test_steps} steps." if done and state == env.goal_state else "Did not reach goal.") 