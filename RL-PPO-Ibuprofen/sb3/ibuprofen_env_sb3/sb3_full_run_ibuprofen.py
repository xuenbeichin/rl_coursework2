import gymnasium as gym
import numpy as np
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class IbuprofenEnv(gym.Env):
    """
    An OpenAI Gym environment simulating the pharmacokinetics of ibuprofen in a human body.

    This environment models the concentration of ibuprofen in the bloodstream after repeated
    doses over a period of time. The agent's goal is to maintain the drug concentration within
    a therapeutic range while avoiding toxicity.

    Attributes:
        action_space (gym.spaces.Discrete): The discrete action space representing dose levels (0 to 4).
        observation_space (gym.spaces.Box): The observation space representing the plasma concentration (0 to 100).
        therapeutic_range (tuple): A tuple defining the lower and upper bounds of the therapeutic range (10 to 50).
        half_life (float): The drug's half-life in hours.
        clearance_rate (float): The clearance rate calculated from the half-life.
        time_step_hours (int): The time step in hours between actions.
        bioavailability (float): Fraction of the drug that reaches systemic circulation (default 0.9).
        volume_of_distribution (float): Volume of distribution in L/kg (default 0.15).
        max_steps (int): Maximum number of steps (24 hours).
        current_step (int): Counter for the current time step in the episode.
        plasma_concentration (float): Current plasma concentration of the drug in µg/mL.
        normalize (bool): Whether to normalize state observations.
        state_buffer (list): A buffer storing past plasma concentrations for normalization.

    Methods:
        reset(seed=None, **kwargs):
            Resets the environment to the initial state.

        step(action):
            Takes a step in the environment with a given action and updates the state.

        _normalize(state):
            Normalizes the state based on past concentrations if normalization is enabled.
    """

    def __init__(self, normalize=False):
        """
        Initializes the Ibuprofen environment.

        Args:
            normalize (bool): Whether to normalize state observations.
        """
        super(IbuprofenEnv, self).__init__()

        # Define the action space: actions represent discrete doses (0-4 units, each unit = 200 mg)
        self.action_space = gym.spaces.Discrete(5)

        # Observation space: plasma concentration of ibuprofen (0-100 µg/mL)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # Pharmacokinetics parameters
        self.therapeutic_range = (10, 50)  # Target therapeutic range for the drug
        self.half_life = 2.0  # Half-life of the drug in hours
        self.clearance_rate = 0.693 / self.half_life  # Clearance rate calculated using half-life
        self.time_step_hours = 1  # Time step in hours
        self.bioavailability = 0.9  # Proportion of the drug absorbed into systemic circulation
        self.volume_of_distribution = 0.15  # Volume of distribution in L/kg
        self.max_steps = 24  # Maximum simulation steps (24 hours)

        # Initial states
        self.current_step = 0
        self.plasma_concentration = 0.0
        self.normalize = normalize
        self.state_buffer = []  # Stores historical plasma concentrations for normalization

    def reset(self, seed=None, **kwargs):
        """
        Resets the environment to the initial state.

        Args:
            seed (int, optional): Random seed for reproducibility.

        Returns:
            tuple: The initial state and an empty info dictionary.
        """
        super().reset(seed=seed)
        self.current_step = 0  # Reset step counter
        self.plasma_concentration = 0.0  # Reset plasma concentration
        self.state_buffer = []  # Clear state buffer
        state = np.array([self.plasma_concentration], dtype=np.float32)  # Initial state
        return self._normalize(state), {}

    def step(self, action):
        """
        Takes a step in the environment based on the selected action.

        Args:
            action (int): The chosen action, representing the dose (0-4 units).

        Returns:
            tuple: Contains the next state, reward, done flag, truncated flag, and info dictionary.
        """
        # Calculate the administered dose (200 mg per unit of action)
        dose_mg = action * 200
        # Calculate the absorbed dose after bioavailability
        absorbed_mg = dose_mg * self.bioavailability
        # Calculate the resulting plasma concentration (µg/mL)
        absorbed_concentration = absorbed_mg / (self.volume_of_distribution * 70)  # Assume 70 kg body weight
        # Update plasma concentration with the absorbed dose
        self.plasma_concentration += absorbed_concentration
        # Apply clearance (exponential decay)
        self.plasma_concentration *= np.exp(-self.clearance_rate * self.time_step_hours)

        # Generate the new state
        state = np.array([self.plasma_concentration], dtype=np.float32)
        normalized_state = self._normalize(state)  # Normalize the state if enabled

        # Append current concentration to state buffer
        self.state_buffer.append(self.plasma_concentration)

        # Calculate reward based on the therapeutic range
        if self.therapeutic_range[0] <= self.plasma_concentration <= self.therapeutic_range[1]:
            reward = 10  # Positive reward for being in the therapeutic range
        else:
            # Penalize for concentrations outside the therapeutic range
            if self.plasma_concentration < self.therapeutic_range[0]:
                reward = -5 - (self.therapeutic_range[0] - self.plasma_concentration) * 0.5
            elif self.plasma_concentration > self.therapeutic_range[1]:
                reward = -5 - (self.plasma_concentration - self.therapeutic_range[1]) * 0.5

        # Apply a severe penalty for toxic concentrations (above 100 µg/mL)
        if self.plasma_concentration > 100:
            reward -= 15

        # Update step counter and check termination conditions
        self.current_step += 1
        done = self.current_step >= self.max_steps  # Episode ends after max steps
        truncated = False  # No explicit truncation condition in this environment
        info = {}  # Additional diagnostics can be added to this dictionary

        return normalized_state, reward, done, truncated, info

    def _normalize(self, state):
        """
        Normalizes the state based on the mean and standard deviation of past concentrations.

        Args:
            state (numpy.ndarray): The current state (plasma concentration).

        Returns:
            numpy.ndarray: The normalized state if normalization is enabled; otherwise, the original state.
        """
        if self.normalize and len(self.state_buffer) > 1:
            mean = np.mean(self.state_buffer)
            std = np.std(self.state_buffer) + 1e-8  # Add a small value to prevent division by zero
            return (state - mean) / std
        return state

from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    def __init__(self):
        super(RewardLoggingCallback, self).__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate reward for the current step
        self.current_episode_reward += self.locals["rewards"][0]

        # If the episode ends, log the reward
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # Reset for the next episode
        return True

def optimize_ppo(trial):
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)
    n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        gamma=gamma,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        batch_size=batch_size,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=0,
    )
    model.learn(total_timesteps=10000)

    rewards = []
    for _ in range(100):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    return np.mean(rewards)

study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=100)

best_params = study.best_params
print("Best Parameters:", best_params)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])

final_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=best_params["learning_rate"],
    gamma=best_params["gamma"],
    n_epochs=best_params["n_epochs"],
    ent_coef=best_params["ent_coef"],
    batch_size=best_params["batch_size"],
    n_steps=best_params["n_steps"],
    gae_lambda=best_params["gae_lambda"],
    clip_range=best_params["clip_range"],
    verbose=1,
)
callback = RewardLoggingCallback()
final_model.learn(total_timesteps=24000, callback=callback)

# Training Loop with Dynamic Time Horizon

# Initialize variables for dynamic time horizon
initial_horizon = 6  # Start with a small time horizon (e.g., 6 hours)
max_horizon = 24     # Full time period (24 hours, in your case)
horizon_increment = 2  # Increase the horizon incrementally
time_horizon = initial_horizon

reward_history = []

# Use a loop for multiple training episodes
for episode in range(1000):  # Number of training episodes
    # Dynamically update the time horizon
    time_horizon = min(max_horizon, initial_horizon + episode * horizon_increment)

    # Reset environment at the beginning of each episode
    state = env.reset()

    total_reward = 0
    plasma_concentration_history = []

    for t in range(time_horizon):  # Use the dynamic time horizon
        # Use the SB3 predict method for actions
        action, _ = final_model.predict(state, deterministic=False)

        # Take a step in the environment
        new_state, reward, done, infos = env.step(action)
        plasma_concentration_history.append(new_state[0])

        # Accumulate the reward
        total_reward += reward

        # Update the current state
        state = new_state

        if done:
            break

    # Append total reward for the episode
    reward_history.append(total_reward)

    # Log progress every 10 episodes
    if episode % 10 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward}, Time Horizon = {time_horizon}")


plt.figure(figsize=(12, 6))
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve (SB3)")
plt.grid()
plt.show()
