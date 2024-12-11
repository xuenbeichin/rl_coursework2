import torch
import numpy as np
from matplotlib import pyplot as plt


def evaluate_agent(env, agent, evaluation_episodes=100):
    """
    Evaluates a trained agent on the environment and collects rewards and state trajectories.

    Args:
        env (gym.Env): The environment to evaluate the agent on.
        agent (PPOAgent): The trained PPO agent with policy and value networks.
        evaluation_episodes (int): Number of episodes to run for evaluation.

    Returns:
        tuple: A tuple containing:
            - evaluation_rewards (list): Total rewards for each evaluation episode.
            - plasma_concentration_trajectories (list): Plasma concentration trajectories for each episode.
    """
    evaluation_rewards = []  # List to store total rewards for each episode
    plasma_concentration_trajectories = []  # List to store plasma concentration trajectories

    for episode in range(evaluation_episodes):
        state, _ = env.reset()  # Reset the environment at the start of each episode
        total_reward = 0
        plasma_concentration_history = [state[0]]  # Track initial plasma concentration

        # Episode loop
        for _ in range(env.max_steps):  # Use max_steps from the environment
            # Convert state to tensor for the policy network
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = agent.policy(state_tensor).detach().numpy().flatten()  # Get action probabilities
            action = np.argmax(action_probs)  # Greedy action selection for evaluation

            # Step in the environment
            new_state, reward, done, truncated, _ = env.step(action)
            plasma_concentration_history.append(new_state[0])  # Track plasma concentration

            state = new_state
            total_reward += reward

            if done or truncated:
                break

        # Store results for this episode
        evaluation_rewards.append(total_reward)
        plasma_concentration_trajectories.append(plasma_concentration_history)

    return evaluation_rewards, plasma_concentration_trajectories

def plot_plasma_concentration_trajectories(env, plasma_concentration_trajectories):
    """
    Plots the plasma concentration trajectory of the last episode.

    This function visualizes the plasma concentration levels over time for the last episode
    in the provided trajectories. It includes annotations for therapeutic ranges and toxic levels
    based on the environment's parameters.

    Args:
        env (gym.Env): The environment used, containing attributes for therapeutic ranges.
        plasma_concentration_trajectories (list of lists): A list where each sublist contains
            plasma concentration values for an episode. The last sublist corresponds to the
            most recent episode's trajectory.

    Returns:
        None: The function displays the plot but does not return any values.

    Visualization Details:
        - The plasma concentration trajectory for the last episode is plotted as a line.
        - Dashed green lines indicate the lower and upper bounds of the therapeutic range.
        - A dashed red line marks the toxic concentration threshold.
        - The x-axis represents time (e.g., in hours), and the y-axis represents plasma concentration.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(plasma_concentration_trajectories[-1], label="Plasma Concentration")
    plt.axhline(y=env.therapeutic_range[0], color="g", linestyle="--", label="Lower Therapeutic Range")
    plt.axhline(y=env.therapeutic_range[1], color="g", linestyle="--", label="Upper Therapeutic Range")
    plt.axhline(y=100, color="r", linestyle="--", label="Toxic Level")
    plt.xlabel("Time (hours)")
    plt.ylabel("Plasma Concentration (mg/L)")
    plt.title("Plasma Concentration Over Time (Custom PPO)")
    plt.legend()
    plt.grid()
    plt.show()
