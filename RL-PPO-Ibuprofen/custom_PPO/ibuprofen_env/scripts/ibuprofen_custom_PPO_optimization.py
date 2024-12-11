import optuna
import numpy as np
import torch
import gymnasium as gym

from custom_PPO.PPOAgent import PPOAgent
from environments.ibuprofen_env import IbuprofenEnv


def optimize_ppo(trial):
    """
    Optimize PPO hyperparameters using Optuna for the Ibuprofen environment.

    This function creates an instance of the Ibuprofen environment, suggests hyperparameters
    using Optuna, trains a PPO agent with the suggested hyperparameters, and evaluates the
    agent based on its performance during training.

    Args:
        trial (optuna.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: The mean reward over the evaluation episodes, used as the objective value.
    """
    # Initialize the custom Ibuprofen environment
    env = IbuprofenEnv(normalize=True)  # Use environment normalization if required

    # Suggest hyperparameters from the search space
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)  # Learning rate
    gamma = trial.suggest_float("gamma", 0.90, 0.99)                 # Discount factor
    eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)             # PPO clipping range
    ppo_epochs = trial.suggest_int("ppo_epochs", 3, 10)              # PPO training epochs
    entropy_beta = trial.suggest_float("entropy_beta", 1e-4, 1e-2, log=True)  # Entropy coefficient
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)   # Batch size
    buffer_size = trial.suggest_int("buffer_size", batch_size * 10, batch_size * 20, step=batch_size)  # Buffer size
    time_horizon = trial.suggest_int("time_horizon", 64, 2048, step=64)  # Time horizon for rollouts
    hidden_units = trial.suggest_int("hidden_units", 32, 512, step=32)  # Hidden layer units
    num_layers = trial.suggest_int("num_layers", 1, 3)                  # Number of layers
    max_steps = trial.suggest_int("max_steps", 500_000, 1_000_000, step=100_000)  # Max training steps
    normalize = trial.suggest_categorical("normalize", [True, False])  # Whether to normalize observations

    # Apply normalization to observation if wanted
    if normalize:
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: (obs - obs.mean()) / (obs.std() + 1e-8),  # Normalize observations
            observation_space=env.observation_space
        )

    # Initialize the PPO agent with the suggested hyperparameters
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],  # Number of state dimensions
        action_dim=env.action_space.n,            # Number of actions
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

    # Training loop for evaluation
    for episode in range(100):  # Train over 100 episodes for evaluation
        states, actions, rewards, dones, old_probs = [], [], [], [], []  # Buffers for episode data
        state, _ = env.reset()  # Reset the environment at the start of the episode
        total_reward = 0  # Initialize the total reward for the episode

        # Run an episode for a maximum of `time_horizon` steps
        for _ in range(time_horizon):
            # Convert state to tensor for policy input
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # Get action probabilities from the agent's policy
            action_probs = agent.policy(state_tensor).detach().numpy()
            # Sample an action based on probabilities
            action = np.random.choice(env.action_space.n, p=action_probs)

            # Take the action in the environment
            new_state, reward, done, truncated, _ = env.step(action)

            # Store experience in the buffers
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done or truncated)
            old_probs.append(action_probs[action])

            # Update state and accumulate reward
            state = new_state
            total_reward += reward

            # End the episode if the environment signals done or truncated
            if done or truncated:
                break

        # Train the agent using the collected episode data
        agent.train((states, actions, rewards, dones, old_probs))

        # Append the total reward for this episode
        reward_history.append(total_reward)

    return np.mean(reward_history)

def get_best_params(n_trials):
    """
    Conducts hyperparameter optimization for the PPO algorithm using Optuna and returns the best parameters.

    This function creates an Optuna study to maximize the mean reward by optimizing PPO hyperparameters.
    It runs the specified number of trials to search for the best combination of hyperparameters.

    Args:
        n_trials (int): The number of trials to run in the hyperparameter optimization process.

    Returns:
        dict: A dictionary containing the best hyperparameters identified by Optuna.
              The dictionary keys correspond to the names of the hyperparameters, and
              the values represent their optimal settings.
    """
    # Create an Optuna study to maximize the mean reward
    study = optuna.create_study(direction="maximize")

    # Run the optimization for the specified number of trials
    study.optimize(optimize_ppo, n_trials=n_trials)

    # Extract and return the best parameters
    best_params = study.best_params
    return best_params
