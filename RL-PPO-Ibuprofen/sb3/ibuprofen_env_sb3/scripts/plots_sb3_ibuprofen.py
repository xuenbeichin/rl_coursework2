from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.ibuprofen_env import IbuprofenEnv


def plot_reward_history(reward_history, title="Learning Curve (SB3)"):
    """
    Plots the reward history to visualize progress over episodes.

    Args:
        reward_history (list): A list of total rewards for each episode.
        title (str, optional): Title of the plot. Defaults to "Learning Curve (SB3)".
    """
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_last_episode_concentration(env, model, therapeutic_range=(10, 50), toxic_threshold=100):
    obs = env.reset()
    plasma_concentrations = []
    done = False
    while not done:
        # Get action from the model
        action, _ = model.predict(obs, deterministic=True)
        # Unpack the four values returned by step
        obs, _, done, info = env.step(action)
        # Access the plasma concentration from the environment
        plasma_concentrations.append(env.envs[0].plasma_concentration)  # Collect plasma concentration

    # Plot the plasma concentration
    plt.figure(figsize=(12, 6))
    plt.plot(plasma_concentrations, label="Plasma Concentration")
    plt.axhline(y=therapeutic_range[0], color='green', linestyle='--', label="Therapeutic Min")
    plt.axhline(y=therapeutic_range[1], color='green', linestyle='--', label="Therapeutic Max")
    plt.axhline(y=toxic_threshold, color='red', linestyle='--', label="Toxic Threshold")
    plt.xlabel("Time Step")
    plt.ylabel("Plasma Concentration")
    plt.title("Plasma Concentration Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


