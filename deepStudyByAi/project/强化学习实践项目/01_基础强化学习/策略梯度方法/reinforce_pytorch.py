import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1) # Output probabilities for each action
        return action_probs

# REINFORCE Agent
class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99):
        self.gamma = gamma
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.policy_network(state_tensor)
        action = torch.multinomial(action_probs, 1).item() # Sample an action from the distribution
        self.log_probs.append(torch.log(action_probs.squeeze(0))[action])
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        G = 0 # Cumulative discounted reward
        returns = []
        for r in self.rewards[::-1]: # Iterate backwards to calculate discounted returns
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        # Normalize returns for more stable training (optional but common)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, Gt in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        self.log_probs = [] # Clear for next episode
        self.rewards = [] # Clear for next episode

# Training loop
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = REINFORCEAgent(state_size, action_size)

    num_episodes = 1000
    scores = []

    for episode in range(num_episodes):
        state = env.reset()[0] # For gym 0.21.0 and later, reset() returns (observation, info)
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action) # For gym 0.21.0 and later, step() returns (observation, reward, terminated, truncated, info)
            agent.store_reward(reward)
            state = next_state
            episode_reward += reward

        agent.learn()
        scores.append(episode_reward)

        # Print average score every 50 episodes
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode + 1}: Average Score = {avg_score:.2f}")

    env.close()
    print("\nTraining finished.")

    # Optional: Test the trained agent
    print("\nTesting the trained agent:")
    test_env = gym.make('CartPole-v1', render_mode='human') # Render for visualization
    state = test_env.reset()[0]
    done = False
    test_reward = 0
    while not done:
        action = agent.choose_action(state) # Agent's policy network will output probabilities
        next_state, reward, done, _, _ = test_env.step(action)
        state = next_state
        test_reward += reward
    test_env.close()
    print(f"Test Episode Total Reward: {test_reward:.2f}") 