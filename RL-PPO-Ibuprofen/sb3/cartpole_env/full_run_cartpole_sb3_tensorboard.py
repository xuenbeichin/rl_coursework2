import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
import optuna
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime


from stable_baselines3.common.callbacks import BaseCallback

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



def optimize_ppo(trial):
    """
    Optimize PPO hyperparameters using Optuna.
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # Wrap the environment in a DummyVecEnv for compatibility with PPO
    env = DummyVecEnv([lambda: env])

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
    model.learn(total_timesteps=100000)


    # Evaluate the agent's performance
    total_rewards = []
    for _ in range(100):
        obs = env.reset()  # No unpacking needed
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)  # Unpack four values instead of five
            total_reward += reward
        total_rewards.append(total_reward)

    # Return the mean of total rewards as the optimization objective
    return np.mean(total_rewards)



def plot_learning_curve(rewards):
    """
    Plot the learning curve using the logged rewards from training.

    Args:
        rewards (list): List of total rewards per episode.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(rewards)), rewards, label="Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve for CartPole-v1 (SB3)")
    plt.legend()
    plt.grid()
    plt.show()


import os
from stable_baselines3.common.logger import configure

def train_and_render_cartpole():
    """
    Train a PPO agent with optimized hyperparameters on CartPole-v1,
    render it using a video recorder, and log training metrics to TensorBoard.
    """
    # Create the environment with render_mode set to 'rgb_array'
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Hyperparameter Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=100)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # Set up TensorBoard logging directory
    log_dir = "./tensorboard/cartpole_ppo"
    os.makedirs(log_dir, exist_ok=True)

    # Configure TensorBoard logger
    new_logger = configure(log_dir, ["tensorboard1"])

    # Train the agent with best hyperparameters
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

    # Set the logger for the PPO model
    final_model.set_logger(new_logger)

    # Train the model with a callback to log rewards
    callback = RewardLoggingCallbackTensor()
    final_model.learn(total_timesteps=100, callback=callback)

    # Plot the learning curve
    plot_learning_curve(callback.episode_rewards)

    # Create the environment
    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Specify the directory and custom name prefix for videos
    video_folder = "./videos"
    video_name_prefix = f"cartpole_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Wrap the environment with RecordVideo
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,  # Record all episodes
        name_prefix=video_name_prefix  # Custom name prefix for the videos
    )

    # Evaluation loop
    state, _ = eval_env.reset()
    done = False
    state_trajectory = []

    while not done:
        state_trajectory.append(state)  # Record the state
        action, _ = final_model.predict(state, deterministic=True)
        state, reward, done, truncated, _ = eval_env.step(action)
        done = done or truncated

    eval_env.close()

    # Plotting Pole Angles
    pole_angles = [s[2] for s in state_trajectory]
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(pole_angles)), pole_angles, label='Pole Angle', color='b')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, label="Vertical Position")
    plt.xlabel('Time Step')
    plt.ylabel('Pole Angle (radians)')
    plt.title('Pole Angle Over Time During Evaluation (SB3)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    train_and_render_cartpole()

