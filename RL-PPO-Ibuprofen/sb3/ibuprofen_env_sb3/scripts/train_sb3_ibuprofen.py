from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3.ibuprofen_env_sb3.scripts.optimize_sb3_ibuprofen import get_best_params
from sb3.ibuprofen_env_sb3.scripts.plots_sb3_ibuprofen import plot_reward_history, plot_last_episode_concentration


def train_ppo_with_callback(env, best_params, total_timesteps, callback):
    """
    Train a PPO model with the provided environment and hyperparameters.

    Args:
        env (DummyVecEnv): The training environment.
        best_params (dict): Dictionary containing optimized hyperparameters.
        total_timesteps (int): Number of timesteps to train the model.
        callback (BaseCallback): Callback for logging or monitoring training.

    Returns:
        PPO: The trained PPO model.
    """
    # Initialize and train the PPO model
    model = PPO(
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
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model


def dynamic_training_loop(model, env, num_episodes, initial_horizon, max_horizon, horizon_increment):
    """
    Run a training loop with a dynamic time horizon.

    Args:
        model (PPO): Trained PPO model.
        env (DummyVecEnv): Environment used for training.
        num_episodes (int): Number of training episodes.
        initial_horizon (int): Initial time horizon for each episode.
        max_horizon (int): Maximum time horizon.
        horizon_increment (int): Increment for time horizon after each episode.

    Returns:
        list: A list of total rewards for each episode.
    """
    reward_history = []

    # Training loop over the specified number of episodes
    for episode in range(num_episodes):
        # Update the time horizon dynamically
        time_horizon = min(max_horizon, initial_horizon + episode * horizon_increment)

        # Reset the environment
        state = env.reset()
        total_reward = 0
        plasma_concentration_history = []

        # Run the episode for the current time horizon
        for t in range(time_horizon):
            # Predict the action using the model
            action, _ = model.predict(state, deterministic=False)

            # Step in the environment
            new_state, reward, done, infos = env.step(action)
            plasma_concentration_history.append(new_state[0])  # Log plasma concentration

            # Accumulate reward
            total_reward += reward

            # Update the state
            state = new_state

            # Break if the episode is done
            if done:
                break

        # Append the total reward for the episode
        reward_history.append(total_reward)

    return reward_history

if __name__ == '__main__':
    from environments.ibuprofen_env import IbuprofenEnv
    from sb3.callback_sb3 import RewardLoggingCallback

    # Define the environment
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])
    callback = RewardLoggingCallback()

    best_params = get_best_params(n_trials=100)

    # Train the PPO model with the callback
    final_model = train_ppo_with_callback(env, best_params, total_timesteps=24000, callback=callback)

    # Run the dynamic training loop
    reward_history = dynamic_training_loop(
        model=final_model,
        env=env,
        num_episodes=1000,
        initial_horizon=6,
        max_horizon=24,
        horizon_increment=2,
    )

    # Plot the reward history
    plot_reward_history(reward_history)

    # Evaluate and plot the plasma concentration for the last episode
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=False)])
    plot_last_episode_concentration(env, final_model)


