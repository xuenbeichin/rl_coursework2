import torch
import numpy as np
import matplotlib.pyplot as plt

def train_agent(agent, env, best_params, episodes=10000):
    """
    Train a PPO agent on the given environment using specified parameters.

    Args:
        agent: The PPO agent to train.
        env: The environment in which the agent will be trained.
        best_params (dict): A dictionary containing the best hyperparameters,
                            including 'buffer_size'.
        episodes (int): Number of episodes for training. Default is 10,000.

    Returns:
        list: A history of total rewards obtained in each episode.
    """
    reward_history = []  # List to store total rewards per episode

    # Training loop over the specified number of episodes
    for episode in range(episodes):
        states, actions, rewards, dones, old_probs = [], [], [], [], []  # Buffers for episode data
        state, _ = env.reset()  # Reset the environment at the start of each episode
        total_reward = 0  # Total reward accumulated in this episode

        # Episode loop: collect data for buffer_size steps or until the episode ends
        for t in range(best_params["buffer_size"]):
            # Convert the state to a tensor for input to the policy network
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # Get action probabilities from the policy network
            action_probs = agent.policy(state_tensor).detach().numpy()
            # Choose an action based on the probabilities
            action = np.random.choice(env.action_space.n, p=action_probs)

            # Perform the action in the environment
            new_state, reward, done, truncated, _ = env.step(action)

            # Store the collected data in the buffers
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done or truncated)
            old_probs.append(action_probs[action])

            # Update the state and accumulate the reward
            state = new_state
            total_reward += reward

            # Break the loop if the episode ends
            if done or truncated:
                break

        # Train the agent using the data collected in this episode
        agent.train((states, actions, rewards, dones, old_probs))

        # Record the total reward for this episode
        reward_history.append(total_reward)

        # Print progress every 50 episodes
        if episode % 50 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return reward_history

def plot_rewards(reward_history):
    """
    Plot the learning curve (total rewards per episode).

    Args:
        reward_history (list): A list of total rewards obtained in each episode.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve for CartPole-v1 (Custom PPO)")
    plt.grid()
    plt.show()
