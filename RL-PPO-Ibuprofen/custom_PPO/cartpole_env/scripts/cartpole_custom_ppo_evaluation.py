import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt


def create_eval_env(video_folder, video_name_prefix):
    """
    Create an evaluation environment with video recording.

    Args:
        video_folder (str): Folder to save video recordings.
        video_name_prefix (str): Prefix for video files.

    Returns:
        eval_env: The environment wrapped with the video recorder.
    """
    # Create the CartPole environment with RGB array rendering mode
    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Wrap the environment to record videos for all episodes
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,  # Record all episodes
        name_prefix=video_name_prefix
    )
    return eval_env


def evaluate_agent(final_model, eval_env):
    """
    Run evaluation on the trained model and return state trajectory.

    Args:
        final_model: The trained RL model to be evaluated.
        eval_env: The evaluation environment to interact with.

    Returns:
        list: A list of states observed during evaluation, representing
              the state trajectory.
    """
    # Reset the environment and get the initial state
    state, _ = eval_env.reset()
    done = False  # Flag to indicate whether the episode has ended
    state_trajectory = []  # To store the trajectory of states

    # Run the evaluation loop
    while not done:
        # Append the current state to the trajectory
        state_trajectory.append(state)

        # Get the action from the model (deterministic policy)
        action, _ = final_model.predict(state, deterministic=True)

        # Step the environment with the chosen action
        state, reward, done, truncated, _ = eval_env.step(action)

        # Check if the episode is done or truncated
        done = done or truncated

    # Close the environment after evaluation
    eval_env.close()

    # Return the trajectory of states
    return state_trajectory


def plot_pole_angles(state_trajectory):
    """
    Plot the pole angles over time from the state trajectory.

    Args:
        state_trajectory (list): A list of states from the environment,
                                 where each state contains the pole angle as the third value.
    """
    # Extract pole angles (third element in each state)
    pole_angles = [s[2] for s in state_trajectory]

    plt.figure(figsize=(12, 6))

    # Plot the pole angles
    plt.plot(range(len(pole_angles)), pole_angles, label='Pole Angle', color='b')

    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, label="Vertical Position")

    plt.xlabel('Time Step')
    plt.ylabel('Pole Angle (radians)')
    plt.title('Pole Angle Over Time During Evaluation (Custom PPO)')
    plt.legend()
    plt.grid()
    plt.show()
