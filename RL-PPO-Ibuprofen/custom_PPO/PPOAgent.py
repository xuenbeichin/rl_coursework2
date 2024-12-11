from custom_PPO.networks import PolicyNetwork, ValueNetwork


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent implementation.

    This class encapsulates the policy and value networks, their optimization, and
    the logic for PPO's training, including computing advantages, value loss, and policy loss.

    Attributes:
        policy (PolicyNetwork): The policy network for action selection.
        value (ValueNetwork): The value network for state value estimation.
        optimizer_policy (torch.optim.Adam): Optimizer for the policy network.
        optimizer_value (torch.optim.Adam): Optimizer for the value network.
        gamma (float): Discount factor for future rewards.
        eps_clip (float): Clipping range for PPO's objective function.
        batch_size (int): Batch size for training.
        ppo_epochs (int): Number of training epochs for PPO updates.
        entropy_beta (float): Coefficient for entropy loss to encourage exploration.
        buffer_size (int): Maximum size of the experience buffer.
        max_steps (int): Maximum number of steps per episode.
        lambda_gae (float): Lambda for Generalized Advantage Estimation (GAE).
    """

    def __init__(self, state_dim, action_dim, lr, gamma, eps_clip, batch_size, ppo_epochs, entropy_beta, buffer_size,
                 max_steps, hidden_units, num_layers, lambda_gae=0.95):
        """
        Initializes the PPO agent with policy and value networks, optimizers, and hyperparameters.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            lr (float): Learning rate for the optimizers.
            gamma (float): Discount factor for rewards.
            eps_clip (float): Clipping range for PPO updates.
            batch_size (int): Batch size for training.
            ppo_epochs (int): Number of PPO update epochs.
            entropy_beta (float): Coefficient for entropy loss.
            buffer_size (int): Maximum size of the experience buffer.
            max_steps (int): Maximum steps per episode.
            hidden_units (int): Number of units in the hidden layers of the networks.
            num_layers (int): Number of layers in the networks.
            lambda_gae (float, optional): GAE lambda for advantage estimation. Defaults to 0.95.
        """
        # Initialize policy and value networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_units, num_layers)
        self.value = ValueNetwork(state_dim, hidden_units, num_layers)

        # Set up optimizers
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)

        # PPO-specific hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.entropy_beta = entropy_beta

        # Additional parameters
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self.lambda_gae = lambda_gae  # For Generalized Advantage Estimation (GAE)

    def compute_advantage(self, rewards, values, dones):
        """
        Compute advantages and discounted rewards using Generalized Advantage Estimation (GAE).

        Args:
            rewards (list or np.ndarray): Rewards for each time step.
            values (list or np.ndarray): Value estimates for each time step.
            dones (list or np.ndarray): Boolean flags indicating the end of episodes.

        Returns:
            np.ndarray: Computed advantages for each time step.
        """
        # Convert inputs to PyTorch tensors
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Append a bootstrap value for the last state
        values = torch.cat([values, torch.tensor([0.0] if dones[-1] else [values[-1]])])

        # Compute deltas (temporal difference errors)
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]

        # Compute advantages using reverse cumulative sum with discount factor
        advantages = torch.zeros_like(deltas)
        discount = self.gamma * self.lambda_gae
        for t in reversed(range(len(deltas))):
            advantages[t] = deltas[t] + (discount * advantages[t + 1] if t + 1 < len(deltas) else 0)

        return advantages.numpy()  # Convert back to NumPy if needed

    def train(self, trajectories):
        """
        Train the PPO agent using collected trajectories.

        Args:
            trajectories (tuple): A tuple containing states, actions, rewards, dones, and old probabilities.

        Steps:
            1. Compute returns and normalize them for stability.
            2. Train the value network to minimize the value loss.
            3. Compute advantages and normalize them.
            4. Train the policy network using PPO's clipped objective and entropy regularization.
        """
        # Unpack the trajectories
        states, actions, rewards, dones, old_probs = map(np.array, trajectories)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)

        # Compute discounted returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + (self.gamma * G * (1 - done))
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns for numerical stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Train the value network
        predicted_values = self.value(states).squeeze()
        value_loss = nn.MSELoss()(predicted_values, returns)  # Mean squared error loss
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # Compute advantages
        advantages = returns - predicted_values.detach()  # Advantage = Return - Value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

        # Optimize the policy network
        for _ in range(self.ppo_epochs):
            for i in range(0, len(states), self.batch_size):
                # Create batches for training
                batch = slice(i, i + self.batch_size)
                state_batch, action_batch, advantage_batch, old_prob_batch = (
                    states[batch], actions[batch], advantages[batch], old_probs[batch])

                # Compute action probabilities
                action_probs = self.policy(state_batch)
                new_probs = action_probs.gather(1, action_batch.unsqueeze(1)).squeeze()
                ratio = new_probs / old_prob_batch

                # Clipped PPO objective
                clip = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss = -torch.min(ratio * advantage_batch, clip * advantage_batch).mean()

                # Entropy loss to encourage exploration
                entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1).mean()
                policy_loss -= self.entropy_beta * entropy_loss

                # Backpropagate and optimize
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()
