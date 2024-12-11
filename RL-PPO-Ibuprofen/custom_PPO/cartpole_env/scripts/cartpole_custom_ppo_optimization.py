import optuna
import numpy as np
import gymnasium as gym
import torch

from custom_PPO.PPOAgent import PPOAgent

def optimize_ppo(trial):
    """
    Optimize PPO hyperparameters using Optuna for the CartPole-v1 environment.

    Args:
        trial (optuna.Trial): An Optuna trial object that suggests hyperparameters.

    Returns:
        float: The mean reward over evaluation episodes, used as the trial's objective value.
    """
    # Create the CartPole environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

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

    # Apply normalization to the observation space if wanted
    if normalize:
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: (obs - obs.mean()) / (obs.std() + 1e-8),  # Normalize observations
            observation_space=env.observation_space
        )

    # Initialize the PPO agent with the suggested hyperparameters
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],  # Number of state dimensions
        action_dim=env.action_space.n,            # Number of possible actions
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

    # Training loop
    for episode in range(50): # Simulate number of episodes
        states, actions, rewards, dones, old_probs = [], [], [], [], []
        state, _ = env.reset()  # Reset environment to start state
        total_reward = 0  # Track total reward for the current episode

        # Run a single episode
        for _ in range(time_horizon):
            # Convert state to a tensor for the policy network
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # Get action probabilities from the policy network
            action_probs = agent.policy(state_tensor).detach().numpy()
            # Sample an action based on the probabilities
            action = np.random.choice(env.action_space.n, p=action_probs)

            # Perform the action in the environment
            new_state, reward, done, truncated, _ = env.step(action)

            # Store experience in buffers
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done or truncated)
            old_probs.append(action_probs[action])

            # Update state and accumulate reward
            state = new_state
            total_reward += reward

            # End the episode if the environment signals done
            if done or truncated:
                break

        # Train the agent with the collected episode data
        agent.train((states, actions, rewards, dones, old_probs))

        # Append total reward for the episode to the history
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
