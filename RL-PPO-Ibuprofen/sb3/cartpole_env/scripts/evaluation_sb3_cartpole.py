from gymnasium.wrappers import RecordVideo
from datetime import datetime

def record_evaluation_video(model, env_id, video_folder, deterministic=True):
    """
    Evaluates a trained RL model and saves the evaluation episode as a video.

    Args:
        model (BaseAlgorithm): The trained RL model to evaluate.
        env_id (str): Gym environment ID (e.g., "CartPole-v1").
        video_folder (str): Folder where videos will be saved.
        deterministic (bool): Whether to use deterministic actions during evaluation.

    Returns:
        None
    """
    eval_env = gym.make(env_id, render_mode="rgb_array")

    video_name_prefix = f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Wrap the environment with RecordVideo to save videos
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,  # Record all episodes
        name_prefix=video_name_prefix  # Custom name prefix for the videos
    )

    # Reset the environment
    state, _ = eval_env.reset()
    done = False

    # Evaluation loop
    while not done:
        # Predict the action using the trained model
        action, _ = model.predict(state, deterministic=deterministic)
        # Step in the environment
        state, reward, done, truncated, _ = eval_env.step(action)
        done = done or truncated  # Treat truncation as episode end

    # Close the environment
    eval_env.close()

    print(f"Video saved in: {video_folder}")


import gym

def evaluate_model(model, env_id, deterministic=True):
    """
    Evaluates a trained RL model and returns the state trajectory.

    Args:
        model (BaseAlgorithm): The trained RL model to evaluate.
        env_id (str): Gym environment ID (e.g., "CartPole-v1").
        deterministic (bool): Whether to use deterministic actions during evaluation.

    Returns:
        list: The trajectory of states observed during the evaluation episode.
    """
    # Create the environment
    eval_env = gym.make(env_id)

    # Reset the environment and initialize variables for evaluation
    state, _ = eval_env.reset()
    done = False
    state_trajectory = []

    # Evaluation loop
    while not done:
        state_trajectory.append(state)  # Record the state trajectory
        action, _ = model.predict(state, deterministic=deterministic)  # Predict action
        state, reward, done, truncated, _ = eval_env.step(action)  # Take a step in the environment
        done = done or truncated  # Handle truncation as episode end

    eval_env.close()  # Close the environment

    return state_trajectory
