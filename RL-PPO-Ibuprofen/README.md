# Implementing the Proximal Policy Optimization (PPO) Algorithm on Ibuprofen Delivery and CartPole-v1

This project simulates the delivery of ibuprofen through a reinforcement learning (RL) agent. The agent interacts with a custom environment that models the pharmacokinetics of ibuprofen in the human body. The goal of the agent is to maintain plasma concentration within a therapeutic range while avoiding toxicity. The project uses **Proximal Policy Optimization (PPO)** for training the agent.

Additionally, the use of CartPole-v1 (an OpenAi's Gym environment) is used for testing PPO implementations.

A PPO algorithm is constructed in the directory ```custom_PPO```, and the directory ```stable_baselines3``` uses stable_baselines3 for the PPO. Moreover, two environments are included (one is a customized drug delivery environment for ibuprofen and CartPole-v1).

**Note**: Each directory has a full .py script and jupyter notebook with a full run of the implementation for demonstration.

## Installation

### 1. Clone the repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/xuenbeichin/RL-PPO-Ibuprofen.git
```

### 2. Install Dependencies
This project requires the following libraries:

numpy: Used for array and matrix operations.
torch: PyTorch for building and training the neural networks.
gym: The OpenAI Gym library for building the environment.
matplotlib: For plotting the results and training performance.

Use pip to install all necessary dependencies listed in requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### 1. stable_baselines3 (SB3) 
The directory _sb3_ is contains the implementation of PPO on the environments ibuprofen and CartPole-v1 using stable_baselines3. 
There are two subfolders for this.  

**_For Ibuprofen Environment_**

**Optimizing Hyperparameters of the PPO Agent**

To optimize the PPO agent:
```python
best_params = get_best_params(n_trials = 100) # Set number of trial episodes
```
**Training the PPO Agent**
```python
env = IbuprofenEnv(normalize=True)
callback = RewardLoggingCallback()

# Train the PPO model with the callback
final_model = train_ppo_with_callback(env, best_params, total_timesteps=24000, callback=callback) # Set total_timesteps

# Run the dynamic training loop to enable the agent to focus on shorter, manageable tasks initially and progressively handle more complex, longer-term strategies as training progresses. Parameters can be customized.
reward_history = dynamic_training_loop(
    model=final_model,
    env=env,
    num_episodes=1000,
    initial_horizon=6,
    max_horizon=24,
    horizon_increment=2,
    )
```
**Plot Learning Curve**
```python
plot_reward_history(reward_history)
```
**Plot Plasma Concentration Over Time For Evaluation**
```python
env = DummyVecEnv([lambda: IbuprofenEnv(normalize=False)])  
plot_last_episode_concentration(env, final_model)
```

**_For CartPole-v1 Environment_**

**Optimizing Hyperparameters of the PPO Agent**

To optimize the PPO agent:
```python
best_params = get_best_params(n_trials = 100) # Set number of trial episodes
```
**Training the PPO Agent**
```python
final_model, callback =  train_and_render_cartpole(best_params, episodes=10000) # Customize number of episodes
```
**Plot Learning Curve**
```python
plot_learning_curve(callback.episode_rewards)
```

**Record Video**
```python
env_id = "CartPole-v1"
video_folder = "/path/to/video"
record_evaluation_video(final_model, env_id, video_folder, deterministic=True)
```

**Plot Pole Trajectory**
```python
state_trajectory = evaluate_model(final_model, env_id, deterministic=True)

plot_pole_angles(state_trajectory)
```

### 2. Custom PPO 
The directory _custom_PPO_ is coded without using any external PPO model. 


**_For Ibuprofen Environment_**
**Optimizing Hyperparameters of the PPO Agent**

To optimize the PPO agent:
```python
best_params = get_best_params(n_trials = 100) # Set number of trial episodes
```

**Training the PPO Agent**

```python
env = IbuprofenEnv(normalize=True)
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=best_params["learning_rate"],
    gamma=best_params["gamma"],
    eps_clip=best_params["eps_clip"],
    batch_size=best_params["batch_size"],
    ppo_epochs=best_params["ppo_epochs"],
    entropy_beta=best_params["entropy_beta"],
    buffer_size=best_params["buffer_size"],
    max_steps=env.max_steps,
    hidden_units=best_params["hidden_units"],
    num_layers=best_params["num_layers"],
)
reward_history = train_ppo_agent(env, agent, num_episodes=10000, initial_horizon=6, max_horizon=24, horizon_increment=2) # Customizable
```

**Plot Learning Curve**
```python
plot_rewards(reward_history)
```

**Plot Plasma Concentration Over Time For Evaluation**
```python
evaluation_rewards, plasma_concentration_trajectories = evaluate_agent(env, agent, evaluation_episodes=100)
plot_plasma_concentration_trajectories(env, plasma_concentration_trajectories)
```


**_For CartPole-v1 Environment_**
**Optimizing Hyperparameters of the PPO Agent**

To optimize the PPO agent:
```python
best_params = get_best_params(n_trials = 100) # Set number of trial episodes
```
**Training the PPO Agent**

```python
env = IbuprofenEnv(normalize=True)
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=best_params["learning_rate"],
    gamma=best_params["gamma"],
    eps_clip=best_params["eps_clip"],
    batch_size=best_params["batch_size"],
    ppo_epochs=best_params["ppo_epochs"],
    entropy_beta=best_params["entropy_beta"],
    buffer_size=best_params["buffer_size"],
    max_steps=env.max_steps,
    hidden_units=best_params["hidden_units"],
    num_layers=best_params["num_layers"],
)
reward_history = train_agent(agent, env, best_params, episodes=10000)
```

**Plot Learning Curve**
```python
plot_rewards(reward_history)
```

**Record CartPole-v1**
```python
video_folder = "/path/to/video"
video_name_prefix = f"cartpole_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
eval_env = create_eval_env(video_folder, video_name_prefix)
```

**Plot Pole Trajectory**
```python
# Evaluate the trained agent in the evaluation environment
state_trajectory = evaluate_agent(final_model, eval_env, episodes)

# Plot the pole angles over time during evaluation
plot_pole_angles(state_trajectory)
```





