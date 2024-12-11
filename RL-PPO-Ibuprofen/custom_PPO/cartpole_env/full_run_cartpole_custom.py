import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium.wrappers import RecordVideo
import optuna
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from datetime import datetime


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, num_layers):
        super(PolicyNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers += [nn.Linear(hidden_units, action_dim), nn.Softmax(dim=-1)]
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        return self.fc(state)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_units, num_layers):
        super(ValueNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers.append(nn.Linear(hidden_units, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        return self.fc(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, eps_clip, batch_size, ppo_epochs, entropy_beta, buffer_size,
                 max_steps, hidden_units, num_layers, lambda_gae=0.95):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_units, num_layers)
        self.value = ValueNetwork(state_dim, hidden_units, num_layers)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.entropy_beta = entropy_beta
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self.lambda_gae = lambda_gae

    def compute_advantage(self, rewards, values, dones):
        """
        Compute advantages and discounted rewards in a fully vectorized manner using PyTorch.
        """
        # Convert everything to PyTorch tensors
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Append bootstrap value for the last state
        values = torch.cat([values, torch.tensor([0.0] if dones[-1] else [values[-1]])])

        # Compute deltas
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]

        # Compute advantages using reverse cumulative sum
        advantages = torch.zeros_like(deltas)
        discount = self.gamma * self.lambda_gae
        for t in reversed(range(len(deltas))):
            advantages[t] = deltas[t] + (discount * advantages[t + 1] if t + 1 < len(deltas) else 0)

        return advantages.numpy()  # Convert back to NumPy for compatibility if needed

    def train(self, trajectories):
        states, actions, rewards, dones, old_probs = map(np.array, trajectories)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)

        # Compute returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + (self.gamma * G * (1 - done))
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize rewards for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute value loss and optimize the value network
        predicted_values = self.value(states).squeeze()
        value_loss = nn.MSELoss()(predicted_values, returns)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # Compute advantages
        advantages = returns - predicted_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize the policy network
        for _ in range(self.ppo_epochs):
            for i in range(0, len(states), self.batch_size):
                batch = slice(i, i + self.batch_size)
                state_batch, action_batch, advantage_batch, old_prob_batch = (
                    states[batch], actions[batch], advantages[batch], old_probs[batch])

                action_probs = self.policy(state_batch)
                new_probs = action_probs.gather(1, action_batch.unsqueeze(1)).squeeze()
                ratio = new_probs / old_prob_batch
                clip = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss = -torch.min(ratio * advantage_batch, clip * advantage_batch).mean()
                entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1).mean()
                policy_loss -= self.entropy_beta * entropy_loss

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()
                self.optimizer_policy.step()



def optimize_ppo(trial):
    # Create the CartPole-v1 environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")  # Set render_mode to rgb_array

    # Define the hyperparameters search space
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)
    ppo_epochs = trial.suggest_int("ppo_epochs", 3, 10)
    entropy_beta = trial.suggest_float("entropy_beta", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)
    buffer_size = trial.suggest_int("buffer_size", batch_size * 10, batch_size * 20, step=batch_size)
    time_horizon = trial.suggest_int("time_horizon", 64, 2048, step=64)
    hidden_units = trial.suggest_int("hidden_units", 32, 512, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    max_steps = trial.suggest_int("max_steps", 500_000, 1_000_000, step=100_000)
    normalize = trial.suggest_categorical("normalize", [True, False])

    # Adjust environment observation normalization
    if normalize:
        env = gym.wrappers.TransformObservation(env, lambda obs: (obs - obs.mean()) / (obs.std() + 1e-8),
                                                observation_space=env.observation_space)
    # Initialize PPO agent with current hyperparameters
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=lr,
        gamma=gamma,
        eps_clip=eps_clip,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        entropy_beta=entropy_beta,
        buffer_size=buffer_size,
        max_steps=max_steps,
        hidden_units=hidden_units,
        num_layers=num_layers,
    )

    reward_history = []  # Track rewards for each episode

    for episode in range(50):
        states, actions, rewards, dones, old_probs = [], [], [], [], []
        state, _ = env.reset()  # Reset the environment
        total_reward = 0

        for _ in range(time_horizon):  # Use time_horizon from search space
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = agent.policy(state_tensor).detach().numpy()
            action = np.random.choice(env.action_space.n, p=action_probs)

            new_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done or truncated)
            old_probs.append(action_probs[action])

            state = new_state
            total_reward += reward

            if done or truncated:
                break

        # Train the agent after each episode with the collected experience
        agent.train((states, actions, rewards, dones, old_probs))

        reward_history.append(total_reward)  # Append the total reward

    # Return mean reward over the episodes
    return np.mean(reward_history)

study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=100)

# Print the Best Hyperparameters
print("Best Hyperparameters:")
print(study.best_params)

# Run the Optuna Optimization
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Train the Agent with the Best Hyperparameters
best_params = study.best_params
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=best_params["learning_rate"],
    gamma=best_params["gamma"],
    eps_clip=best_params["eps_clip"],
    batch_size=best_params["batch_size"],
    ppo_epochs=best_params["ppo_epochs"],
    entropy_beta=best_params["entropy_beta"],
    buffer_size=best_params["buffer_size"],  # Include buffer_size
    max_steps=best_params["max_steps"],      # Include max_steps
    hidden_units=best_params["hidden_units"],  # Include hidden_units
    num_layers=best_params["num_layers"],    # Include num_layers
)


reward_history = []
episodes = 10000

for episode in range(episodes):
    states, actions, rewards, dones, old_probs = [], [], [], [], []
    state, _ = env.reset()
    total_reward = 0

    for t in range(best_params["buffer_size"]):  # Use time_horizon from best_params here
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = agent.policy(state_tensor).detach().numpy()
        action = np.random.choice(env.action_space.n, p=action_probs)

        new_state, reward, done, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done or truncated)
        old_probs.append(action_probs[action])

        state = new_state
        total_reward += reward

        if done or truncated:
            break

    agent.train((states, actions, rewards, dones, old_probs))
    reward_history.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Plot Learning Curve
plt.figure(figsize=(12, 6))
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve for CartPole-v1 (Custom PPO)")
plt.grid()
plt.show()

# Specify the video folder and a unique prefix for this run
video_folder = "/Users/xuenbei/Desktop/rl_coursework2_02015483/custom_PPO/cartpole_env/videos"
video_name_prefix = f"cartpole_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create the environment with RecordVideo wrapper
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
eval_env = RecordVideo(
    eval_env,
    video_folder=video_folder,
    episode_trigger=lambda e: True,  # Record all episodes
    name_prefix=video_name_prefix   # Custom name prefix
)

# Run evaluation
state, _ = eval_env.reset()
done = False
state_trajectory = []

while not done:
    state_trajectory.append(state)  # Record the state for pole angle analysis

    # Convert state to tensor and get action probabilities
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Ensure batch dimension
    action_probs = agent.policy(state_tensor).detach().numpy()

    # Choose an action (stochastic sampling or deterministic)
    action = np.argmax(action_probs.squeeze())  # Deterministic action

    # Step in the environment
    state, reward, done, truncated, _ = eval_env.step(action)

    # Check if it's a terminal state
    done = done or truncated

# Close the environment after recording
eval_env.close()

# Extract Pole Angles from State Trajectory
pole_angles = [s[2] for s in state_trajectory]

# Plot Pole Angle Over Time During Evaluation
plt.figure(figsize=(12, 6))
plt.plot(range(len(pole_angles)), pole_angles, label='Pole Angle', color='b')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, label="Vertical Position")
plt.xlabel('Time Step')
plt.ylabel('Pole Angle (radians)')
plt.title('Pole Angle Over Time During Evaluation (Custom PPO)')
plt.legend()
plt.grid()
plt.show()

# Output the location of the video
print(f"Video saved at: {video_folder}")