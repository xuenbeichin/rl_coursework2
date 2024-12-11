import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

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


class RewardLoggingCallbackTensor(BaseCallback):
    """
    Custom callback for logging rewards and additional PPO-specific metrics to TensorBoard.

    This callback tracks episode rewards during training and logs them to TensorBoard. It also
    records key PPO-specific metrics, such as policy loss and explained variance, at the end
    of each rollout.

    Attributes:
        episode_rewards (list): List of total rewards for completed episodes.
        current_episode_reward (float): Accumulated reward for the current episode.
        episode_count (int): Counter for the number of completed episodes.
    """

    def __init__(self):
        """
        Initializes the RewardLoggingCallback.

        Sets up the episode reward tracker and initializes counters for reward tracking.
        """
        super(RewardLoggingCallbackTensor, self).__init__()
        self.episode_rewards = []  # Store rewards for each episode
        self.current_episode_reward = 0  # Accumulate rewards for the current episode
        self.episode_count = 0  # Count the number of completed episodes

    def _on_step(self) -> bool:
        """
        Called after each environment step during training.

        Accumulates the reward for the current episode and logs it when the episode ends.

        Returns:
            bool: Always returns True, allowing the training process to continue.
        """
        # Add the reward from the current step to the episode total
        self.current_episode_reward += self.locals["rewards"][0]

        # Check if the episode has ended
        if self.locals["dones"][0]:
            # Log the total reward for the episode to TensorBoard
            self.logger.record("episode/reward", self.current_episode_reward)
            # Append the episode reward to the list
            self.episode_rewards.append(self.current_episode_reward)
            # Increment the episode count
            self.episode_count += 1
            # Reset the reward accumulator for the next episode
            self.current_episode_reward = 0

        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of a rollout.

        Logs additional PPO-specific metrics such as policy loss, value loss, entropy,
        and explained variance to TensorBoard.
        """
        try:
            # Define the PPO-specific metrics to log
            metrics = {
                "train/approxkl": self.locals.get("approx_kl", None),  # Approximate KL divergence
                "train/clipfrac": self.locals.get("clip_fraction", None),  # Clip fraction
                "train/entropy_loss": self.locals.get("entropy_loss", None),  # Entropy loss
                "train/explained_variance": self.locals.get("explained_variance", None),  # Explained variance
                "train/fps": self.locals.get("fps", None),  # Frames per second
                "train/policy_loss": self.locals.get("policy_loss", None),  # Policy loss
                "train/value_loss": self.locals.get("value_loss", None),  # Value loss
                "train/loss": self.locals.get("loss", None),  # Total loss
            }

            # Log each metric to TensorBoard if available
            for key, value in metrics.items():
                if value is not None:
                    self.logger.record(key, value)
        except KeyError:
            # If some metrics are unavailable, skip logging
            pass

class TensorBoardCallback(BaseCallback):
    """
    Custom callback for logging training metrics to TensorBoard.

    This callback tracks rewards and key PPO-specific metrics during training and logs them
    to TensorBoard. It supports dynamic reward tracking and logs metrics such as:
        - Approximate KL divergence
        - Clip fraction
        - Policy entropy
        - Policy loss
        - Value loss
        - Explained variance

    Attributes:
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer instance for logging.
        episode_rewards (list): List of total rewards for completed episodes.
        current_episode_reward (float): Accumulated reward for the current episode.
    """

    def __init__(self, writer):
        """
        Initializes the TensorBoardCallback.

        Args:
            writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging.
        """
        super(TensorBoardCallback, self).__init__()
        self.writer = writer  # TensorBoard writer instance
        self.episode_rewards = []  # List to store episode rewards
        self.current_episode_reward = 0  # Accumulate rewards for the current episode

    def _on_step(self) -> bool:
        """
        Called after each environment step during training.

        Tracks the reward for the current episode and logs it to TensorBoard when the episode ends.
        Logs additional training metrics after every step.

        Returns:
            bool: Always returns True, allowing the training process to continue.
        """
        # Accumulate the reward for the current step
        self.current_episode_reward += self.locals["rewards"][0]

        # Check if the episode has ended
        if self.locals["dones"][0]:
            # Log the total reward for the episode
            episode = len(self.episode_rewards)
            self.episode_rewards.append(self.current_episode_reward)
            self.writer.add_scalar("Reward/Episode", self.current_episode_reward, episode)
            # Reset the reward accumulator for the next episode
            self.current_episode_reward = 0

        # Log additional PPO-specific metrics to TensorBoard
        self.writer.add_scalar("Metrics/approxkl", self.locals.get("approx_kl", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/clipfrac", self.locals.get("clipfrac", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/explained_variance", self.locals.get("explained_variance", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/policy_entropy", self.locals.get("entropy", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/policy_loss", self.locals.get("pg_loss", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/value_loss", self.locals.get("value_loss", 0), self.num_timesteps)

        return True

    def _on_training_end(self):
        """
        Called at the end of training.

        Ensures all metrics are properly written and closes the TensorBoard writer.
        """
        self.writer.close()



def optimize_ppo(trial):
    """
    Optimize PPO hyperparameters using Optuna.
    """
    # Environment
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])

    # Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)
    n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    # Model
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

    # Train the model for a sufficient number of timesteps
    model.learn(total_timesteps=10000)  # Increased training timesteps

    # Evaluation
    rewards = []
    for _ in range(10):  #
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    # Return the mean reward over evaluation episodes
    return np.mean(rewards)


# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=100)

best_params = study.best_params
print("Best Parameters:", best_params)

# TensorBoard logging directory
log_dir = "./tensorboard/ppo_ibuprofen"
new_logger = configure(log_dir, ["tensorboard"])

# Final model training
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
final_model.set_logger(new_logger)

# Train with custom callback
callback = RewardLoggingCallbackTensor()
final_model.learn(total_timesteps=10000, callback=callback)
