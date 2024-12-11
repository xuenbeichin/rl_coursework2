import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch.nn.functional as F


class IbuprofenEnv(gym.Env):
    def __init__(self, normalize=False):
        super(IbuprofenEnv, self).__init__()

        # Define the action space and observation space
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # Pharmacokinetics parameters
        self.therapeutic_range = (10, 50)
        self.half_life = 2.0
        self.clearance_rate = 0.693 / self.half_life
        self.time_step_hours = 1
        self.bioavailability = 0.9
        self.volume_of_distribution = 0.15
        self.max_steps = 24
        self.current_step = 0
        self.plasma_concentration = 0.0
        self.state_buffer = []
        self.normalize = normalize

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.current_step = 0
        self.plasma_concentration = 0.0
        self.state_buffer = []
        state = np.array([self.plasma_concentration], dtype=np.float32)
        return self._normalize(state), {}

    def step(self, action):
        # Dose administration
        dose_mg = action * 200
        absorbed_mg = dose_mg * self.bioavailability
        absorbed_concentration = absorbed_mg / (self.volume_of_distribution * 70)
        self.plasma_concentration += absorbed_concentration
        self.plasma_concentration *= np.exp(-self.clearance_rate * self.time_step_hours)

        state = np.array([self.plasma_concentration], dtype=np.float32)
        normalized_state = self._normalize(state)

        self.state_buffer.append(self.plasma_concentration)

        # Reward shaping logic
        if self.therapeutic_range[0] <= self.plasma_concentration <= self.therapeutic_range[1]:
            # Positive reward for being in the therapeutic range
            reward = 10
        else:
            # Penalty for being below or above the therapeutic range
            if self.plasma_concentration < self.therapeutic_range[0]:
                penalty = (self.therapeutic_range[0] - self.plasma_concentration) * 0.1
                reward = -5 - penalty
            elif self.plasma_concentration > self.therapeutic_range[1]:
                penalty = (self.plasma_concentration - self.therapeutic_range[1]) * 0.1
                reward = -5 - penalty

        # Add gradual penalty for instability (fluctuations in concentration)
        if len(self.state_buffer) > 1:
            fluctuation_penalty = abs(self.state_buffer[-1] - self.state_buffer[-2]) * 0.05
            reward -= fluctuation_penalty

        # Add a heavy penalty for toxic concentrations
        if self.plasma_concentration > 100:
            reward -= 15  # Severe penalty for toxic levels

        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False  # For this environment, there's no specific truncation condition
        info = {}  # Additional diagnostic information

        return normalized_state, reward, done, truncated, info

    def _normalize(self, state):
        if self.normalize and len(self.state_buffer) > 1:
            mean = np.mean(self.state_buffer)
            std = np.std(self.state_buffer) + 1e-8
            return (state - mean) / std
        return state

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
        self.lambda_gae = lambda_gae  # For GAE

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



def objective(trial):
    # Define the search space for time_horizon
    time_horizon = trial.suggest_int("time_horizon", 6, 24, step=2)  # Tune from 6 to 24 in steps of 2

    # Other hyperparameters to optimize
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)
    ppo_epochs = trial.suggest_int("ppo_epochs", 3, 10)
    entropy_beta = trial.suggest_float("entropy_beta", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)
    buffer_size = trial.suggest_int("buffer_size", batch_size * 10, batch_size * 20, step=batch_size)
    hidden_units = trial.suggest_int("hidden_units", 32, 512, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    normalize = trial.suggest_categorical("normalize", [True, False])

    # Initialize environment and PPO agent
    env = IbuprofenEnv(normalize=normalize)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=lr,
        gamma=gamma,
        eps_clip=eps_clip,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        max_steps=env.max_steps,
        entropy_beta=entropy_beta,
        buffer_size=buffer_size,
        hidden_units=hidden_units,
        num_layers=num_layers,
    )

    # Training loop for Optuna
    reward_history = []

    for episode in range(100):  # Number of episodes per trial
        states, actions, rewards, dones, old_probs = [], [], [], [], []
        state, _ = env.reset()
        total_reward = 0

        for t in range(time_horizon):  # Use the sampled `time_horizon`
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

        # Train PPO with collected data (include `episode`)
        agent.train((states, actions, rewards, dones, old_probs), episode)
        reward_history.append(total_reward)

    # Return mean reward over episodes for Optuna to optimize
    return np.mean(reward_history)

# Run the Optuna Optimization
env = IbuprofenEnv(normalize=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Print the Best Hyperparameters
print("Best Hyperparameters:")
print(study.best_params)


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
    buffer_size=best_params["buffer_size"],
    max_steps=env.max_steps,
    hidden_units=best_params["hidden_units"],
    num_layers=best_params["num_layers"],
)



buffer_size = best_params["buffer_size"]  # From hyperparameter optimization
batch_size = best_params["batch_size"]

# Initialize variables for dynamic time horizon, so the agent can learn short-term policies initially.
# Gradually adapt to long-term dependencies, and stabilize training by reducing variance in early rewards.
initial_horizon = 6  # Start with a small time horizon
max_horizon = 24     # Full time period (24 hours)
horizon_increment = 2  # Increase the horizon incrementally
time_horizon = initial_horizon

# Training loop
reward_history = []

for episode in range(10000):  # Number of training episodes

    time_horizon = min(max_horizon, initial_horizon + episode * horizon_increment)

    states, actions, rewards, dones, old_probs = [], [], [], [], []
    state, _ = env.reset()  # Reset environment at the start of the episode
    total_reward = 0

    for t in range(time_horizon):  # Use the dynamic time horizon instead of fixed `max_steps`
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = agent.policy(state_tensor).detach().numpy()
        action = np.random.choice(env.action_space.n, p=action_probs)

        new_state, reward, done, truncated, _ = env.step(action)

        # Store transitions
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done or truncated)
        old_probs.append(action_probs[action])

        state = new_state
        total_reward += reward

        if done or truncated:
            break

    # Train PPO with collected trajectories
    agent.train((states, actions, rewards, dones, old_probs), episode)  # Pass episode as an argument
    reward_history.append(total_reward)


    if episode % 50 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward}, Time Horizon = {time_horizon}")



# Plot rewards
plt.figure(figsize=(12, 6))
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve (Custom PPO)")
plt.grid()
plt.show()


# Evaluation Loop
evaluation_episodes = 100  # Number of episodes for evaluation
state, _ = env.reset()

evaluation_rewards = []
plasma_concentration_trajectories = []

for episode in range(evaluation_episodes):
    state, _ = env.reset()
    total_reward = 0
    plasma_concentration_history = [state[0]]  # Track plasma concentration

    for _ in range(env.max_steps):  # Use max_steps from environment
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = agent.policy(state_tensor).detach().numpy().flatten()
        action = np.argmax(action_probs)  # Greedy action selection for evaluation

        new_state, reward, done, truncated, _ = env.step(action)
        plasma_concentration_history.append(new_state[0])

        state = new_state
        total_reward += reward

        if done or truncated:
            break

    evaluation_rewards.append(total_reward)
    plasma_concentration_trajectories.append(plasma_concentration_history)

# Plot plasma concentration from the last evaluation episode
plt.figure(figsize=(12, 6))
plt.plot(plasma_concentration_trajectories[-1], label="Plasma Concentration")
plt.axhline(y=env.therapeutic_range[0], color="g", linestyle="--", label="Lower Therapeutic Range")
plt.axhline(y=env.therapeutic_range[1], color="g", linestyle="--", label="Upper Therapeutic Range")
plt.axhline(y=100, color="r", linestyle="--", label="Toxic Level")
plt.xlabel("Time (hours)")
plt.ylabel("Plasma Concentration (mg/L)")
plt.title("Plasma Concentration Over Time (Custom PPO)")
plt.legend()
plt.grid()
plt.show()


