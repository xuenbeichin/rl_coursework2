import torch
import numpy as np
from matplotlib import pyplot as plt

from custom_PPO.PPOAgent import PPOAgent


def create_ppo_agent(best_params, state_dim, action_dim, max_steps):
    """
    Initializes and returns a PPOAgent using the best hyperparameters.

    Args:
        best_params (dict): Dictionary containing optimized hyperparameters for the PPO agent.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        max_steps (int): Maximum number of steps per episode in the environment.

    Returns:
        PPOAgent: The initialized PPO agent.
    """
    # Create and return the PPOAgent with the best hyperparameters
    return PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=best_params["learning_rate"],
        gamma=best_params["gamma"],
        eps_clip=best_params["eps_clip"],
        batch_size=best_params["batch_size"],
        ppo_epochs=best_params["ppo_epochs"],
        entropy_beta=best_params["entropy_beta"],
        buffer_size=best_params["buffer_size"],
        max_steps=max_steps,
        hidden_units=best_params["hidden_units"],
        num_layers=best_params["num_layers"],
    )

def train_ppo_agent(env, agent, num_episodes=10000, initial_horizon=6, max_horizon=24, horizon_increment=2):
    """
    Train a PPO agent using the best hyperparameters with a dynamic time horizon.

    Args:
        env (gym.Env): The environment for training the agent.
        agent (PPOAgent): The PPO agent to train.
        num_episodes (int, optional): Number of training episodes. Defaults to 10000.
        initial_horizon (int, optional): Starting time horizon for training. Defaults to 6.
        max_horizon (int, optional): Maximum time horizon. Defaults to 24.
        horizon_increment (int, optional): Increment for the time horizon after each episode. Defaults to 2.

    Returns:
        list: A list of total rewards collected during training.
    """

    reward_history = []

    for episode in range(num_episodes):
        # Dynamically adjust the time horizon
        time_horizon = min(max_horizon, initial_horizon + episode * horizon_increment)

        # Initialize storage for trajectories
        states, actions, rewards, dones, old_probs = [], [], [], [], []
        state, _ = env.reset()  # Reset environment at the start of the episode
        total_reward = 0

        for t in range(time_horizon):
            # Convert state to tensor for the policy network
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = agent.policy(state_tensor).detach().numpy()  # Get action probabilities
            action = np.random.choice(env.action_space.n, p=action_probs)  # Sample an action

            # Take a step in the environment
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

        # Train the PPO agent with the collected trajectories
        agent.train((states, actions, rewards, dones, old_probs), episode)
        reward_history.append(total_reward)  # Record total reward for the episode

    return reward_history


def plot_rewards(reward_history):
    """
    Plot the total rewards per episode to visualize the agent's learning curve.

    Args:
        reward_history (list): A list of total rewards obtained in each episode during training.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve (Custom PPO)")
    plt.grid()
    plt.legend()
    plt.show()
