from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggingCallback(BaseCallback):
    """
    Custom callback for logging episode rewards during training.

    This callback tracks rewards for each episode during training. It accumulates rewards
    step-by-step and logs the total reward at the end of each episode. The rewards for all
    completed episodes are stored in a list for analysis or visualization after training.

    Attributes:
        episode_rewards (list): A list of total rewards for all completed episodes.
        current_episode_reward (float): The cumulative reward for the current episode.
    """

    def __init__(self):
        """
        Initializes the RewardLoggingCallback.

        Sets up the storage for episode rewards and initializes the current episode reward tracker.
        """
        super(RewardLoggingCallback, self).__init__()
        self.episode_rewards = []  # List to store rewards for completed episodes
        self.current_episode_reward = 0  # Accumulates rewards for the current episode

    def _on_step(self) -> bool:
        """
        Called after each environment step during training.

        Accumulates the reward for the current step into the current episode reward.
        If the episode ends, logs the total episode reward and resets the tracker.

        Returns:
            bool: Always returns True, allowing training to continue.
        """
        # Accumulate reward for the current step
        self.current_episode_reward += self.locals["rewards"][0]

        # Check if the episode is done
        if self.locals["dones"][0]:
            # Log the total reward for the completed episode
            self.episode_rewards.append(self.current_episode_reward)
            # Reset the tracker for the next episode
            self.current_episode_reward = 0

        return True



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
