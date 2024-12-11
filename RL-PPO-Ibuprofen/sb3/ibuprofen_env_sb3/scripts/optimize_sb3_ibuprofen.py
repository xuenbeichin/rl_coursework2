import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.ibuprofen_env import IbuprofenEnv


def optimize_ppo(trial):
    """
    Optimize hyperparameters for the PPO algorithm using Optuna.

    This function initializes the environment, suggests hyperparameters for PPO using Optuna,
    trains the PPO model, and evaluates its performance over multiple episodes. The mean reward
    across the evaluation episodes is returned as the objective function value for Optuna to optimize.

    Args:
        trial (optuna.trial.Trial): A trial object that suggests hyperparameter values.

    Returns:
        float: The mean reward obtained from the evaluation episodes, representing the
               performance of the PPO agent with the suggested hyperparameters.
    """
    # Create a vectorized environment with normalization enabled
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])

    # Suggest hyperparameters for PPO to try for optimization
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)  # Learning rate
    gamma = trial.suggest_float("gamma", 0.90, 0.99)  # Discount factor
    n_epochs = trial.suggest_int("n_epochs", 3, 10)  # Number of training epochs per update
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)  # Entropy coefficient
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)  # Batch size
    n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)  # Number of steps per rollout
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)  # GAE lambda for advantage calculation
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)  # Clipping range for PPO updates

    # Initialize the PPO model with the suggested hyperparameters
    model = PPO(
        "MlpPolicy",  # Use a multi-layer perceptron policy
        env,  # Training environment
        learning_rate=lr,  # Learning rate
        gamma=gamma,  # Discount factor
        n_epochs=n_epochs,  # Number of training epochs
        ent_coef=ent_coef,  # Entropy coefficient
        batch_size=batch_size,  # Batch size
        n_steps=n_steps,  # Rollout steps
        gae_lambda=gae_lambda,  # GAE lambda
        clip_range=clip_range,  # Clipping range
        verbose=0,  # Suppress output for Optuna optimization
    )

    # Train the PPO model for a predefined number of timesteps
    model.learn(total_timesteps=10000)

    # Evaluate the trained model over multiple episodes
    rewards = []  # List to store total rewards for each episode
    for _ in range(100):  # Evaluate over 100 episodes
        obs = env.reset()  # Reset the environment
        total_reward = 0  # Initialize total reward for the episode
        done = False  # Track if the episode is complete
        while not done:
            # Predict the action using the trained model
            action, _ = model.predict(obs, deterministic=True)
            # Take a step in the environment with the chosen action
            obs, reward, done, _ = env.step(action)
            total_reward += reward  # Accumulate the reward
        rewards.append(total_reward)  # Store the total reward for this episode

    # Return the mean reward across evaluation episodes
    return np.mean(rewards)

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

