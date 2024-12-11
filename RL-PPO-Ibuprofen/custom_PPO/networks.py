import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    A neural network for the policy in reinforcement learning.

    This network outputs a probability distribution over actions based on the current state,
    using a multi-layer feedforward architecture with ReLU activations and a final softmax layer.

    Attributes:
        fc (nn.Sequential): The sequential feedforward layers of the network.
    """

    def __init__(self, state_dim, action_dim, hidden_units, num_layers):
        """
        Initializes the PolicyNetwork.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the action space (number of possible actions).
            hidden_units (int): Number of units in each hidden layer.
            num_layers (int): Number of layers in the network (excluding input and output layers).
        """
        super(PolicyNetwork, self).__init__()

        # Create the layers for the policy network
        # First layer maps state_dim to hidden_units
        layers = [nn.Linear(state_dim, hidden_units), nn.ReLU()]

        # Add additional hidden layers as specified by num_layers
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]

        # Final layer maps to action_dim and applies softmax for probability output
        layers += [nn.Linear(hidden_units, action_dim), nn.Softmax(dim=-1)]

        # Combine all layers into a sequential model
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass through the policy network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: A tensor representing the action probabilities.
        """
        return self.fc(state)


class ValueNetwork(nn.Module):
    """
    A neural network for estimating the state value in reinforcement learning.

    This network predicts the expected return (value) of a given state, using a
    multi-layer feedforward architecture with ReLU activations and a final linear layer.

    Attributes:
        fc (nn.Sequential): The sequential feedforward layers of the network.
    """

    def __init__(self, state_dim, hidden_units, num_layers):
        """
        Initializes the ValueNetwork.

        Args:
            state_dim (int): Dimension of the input state.
            hidden_units (int): Number of units in each hidden layer.
            num_layers (int): Number of layers in the network (excluding input and output layers).
        """
        super(ValueNetwork, self).__init__()

        # Create the layers for the value network
        # First layer maps state_dim to hidden_units
        layers = [nn.Linear(state_dim, hidden_units), nn.ReLU()]

        # Add additional hidden layers as specified by num_layers
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]

        # Final layer outputs a single scalar value
        layers.append(nn.Linear(hidden_units, 1))

        # Combine all layers into a sequential model
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass through the value network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: A tensor representing the estimated state value.
        """
        return self.fc(state)
