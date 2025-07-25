import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

# Define the Actor Network (Policy Network)
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs

# Define the Critic Network (Value Network)
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1) # Output a single value (V(s))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        value = self.fc2(x)
        return value

# Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.gamma = gamma

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.log_probs = []
        self.rewards = []
        self.states = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.actor(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        self.log_probs.append(torch.log(action_probs.squeeze(0))[action])
        self.states.append(state)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self, next_state, done):
        # Convert stored data to tensors
        states_tensor = torch.FloatTensor(np.array(self.states))
        log_probs_tensor = torch.cat(self.log_probs)

        # Calculate returns (G_t)
        G = 0
        returns = []
        for r in self.rewards[::-1]:
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_tensor = torch.FloatTensor(returns)

        # Compute value estimates for current states and next state
        current_values = self.critic(states_tensor).squeeze()
        next_state_tensor = torch.FloatTensor(next_state)
        next_value = self.critic(next_state_tensor).squeeze()

        # If the episode is done, next_value is 0
        if done:
            target_values = returns_tensor
        else:
            # TD Target: r + gamma * V(s')
            td_targets = torch.FloatTensor(self.rewards) + self.gamma * next_value
            # For the last state in the trajectory, we need to handle its TD target carefully.
            # For all states but the last, use TD_target = reward + gamma * V(s_next)
            # For the last state, use its actual return as the target for the critic update.
            # A simpler way is to just use returns as targets for critic, and TD error for actor.
            # Let's use the TD error for both for simplicity and standard A2C.
            # The returns calculated above are actually Monte Carlo returns, which can be noisy.
            # For A2C, we use TD error as the advantage.
            # Advantage = r + gamma * V(s') - V(s)
            # We'll use the returns (Monte Carlo estimate) for the critic loss here as a simplification
            # of the most basic Actor-Critic, where critic learns V(s) by minimizing MSE with Monte Carlo returns.
            # For more advanced AC (like A2C), the advantage calculation is more precise.
            target_values = returns_tensor # This is effectively Monte Carlo return as critic target

        # Critic Loss (MSE between predicted values and targets)
        critic_loss = F.mse_loss(current_values, target_values)

        # Calculate advantages (TD error or G_t - V(s))
        # Using Monte Carlo returns for advantage for simplicity here.
        # A more common approach is TD_target - V(s)
        advantages = returns_tensor - current_values.detach() # Detach to prevent gradients flowing to critic from actor loss

        # Actor Loss (Policy Gradient with advantage)
        actor_loss = (-log_probs_tensor * advantages).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Clear memory for next episode
        self.log_probs = []
        self.rewards = []
        self.states = []

# Training loop
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = ActorCriticAgent(state_size, action_size)

    num_episodes = 1000
    scores = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
            episode_reward += reward

        agent.learn(next_state, done)
        scores.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode + 1}: Average Score = {avg_score:.2f}")

    env.close()
    print("\nTraining finished.")

    # Optional: Test the trained agent
    print("\nTesting the trained agent:")
    test_env = gym.make('CartPole-v1', render_mode='human')
    state = test_env.reset()[0]
    done = False
    test_reward = 0
    while not done:
        with torch.no_grad(): # No need to track gradients during testing
            state_tensor = torch.FloatTensor(state)
            action_probs = agent.actor(state_tensor)
            action = torch.argmax(action_probs).item() # Choose deterministic action for testing
        
        next_state, reward, done, _, _ = test_env.step(action)
        state = next_state
        test_reward += reward
    test_env.close()
    print(f"Test Episode Total Reward: {test_reward:.2f}") 