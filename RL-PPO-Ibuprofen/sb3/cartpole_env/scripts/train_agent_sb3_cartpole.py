from matplotlib import pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
from sb3.callback_sb3 import RewardLoggingCallback

def train_and_render_cartpole(best_params, episodes=10000):
    """
    Train a PPO agent with optimized hyperparameters on CartPole-v1
    and render it using a video recorder.

    Args:
        best_params (dict): The best hyperparameters from Optuna optimization.
        episodes (int): Number of episodes to train the agent.

    Returns:
        final_model: The trained PPO model.
        callback: The reward logging callback used during training.
    """
    # Create the environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

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
    callback = RewardLoggingCallback()
    final_model.learn(total_timesteps=episodes * best_params["n_steps"], callback=callback)

    return final_model, callback

def plot_learning_curve(rewards):
    """
    Plots the learning curve showing rewards per episode during training.

    Args:
        rewards (list or array-like): A list of rewards logged during training.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Learning Curve for CartPole-v1 (SB3)")
    plt.grid()
    plt.show()

