import gymnasium as gym
import numpy as np
import numpy as np


class IbuprofenEnv(gym.Env):
    """
    An OpenAI Gym environment simulating the pharmacokinetics of ibuprofen in a human body.

    This environment models the concentration of ibuprofen in the bloodstream after repeated
    doses over a period of time. The agent's goal is to maintain the drug concentration within
    a therapeutic range while avoiding toxicity.

    Attributes:
        action_spxqace (gym.spaces.Discrete): The discrete action space representing dose levels (0 to 4).
        observation_space (gym.spaces.Box): The observation space representing the plasma concentration (0 to 100).
        therapeutic_range (tuple): A tuple defining the lower and upper bounds of the therapeutic range (10 to 50).
        half_life (float): The drug's half-life in hours.
        clearance_rate (float): The clearance rate calculated from the half-life.
        time_step_hours (int): The time step in hours between actions.
        bioavailability (float): Fraction of the drug that reaches systemic circulation (default 0.9).
        volume_of_distribution (float): Volume of distribution in L/kg (default 0.15).
        max_steps (int): Maximum number of steps (24 hours).
        current_step (int): Counter for the current time step in the episode.
        plasma_concentration (float): Current plasma concentration of the drug in µg/mL.
        normalize (bool): Whether to normalize state observations.
        state_buffer (list): A buffer storing past plasma concentrations for normalization.

    Methods:
        reset(seed=None, **kwargs):
            Resets the environment to the initial state.

        step(action):
            Takes a step in the environment with a given action and updates the state.

        _normalize(state):
            Normalizes the state based on past concentrations if normalization is enabled.
    """

    def __init__(self, normalize=False):
        """
        Initializes the Ibuprofen environment.

        Args:
            normalize (bool): Whether to normalize state observations.
        """
        super(IbuprofenEnv, self).__init__()

        # Define the action space: actions represent discrete doses (0-4 units, each unit = 200 mg)
        self.action_space = gym.spaces.Discrete(5)

        # Observation space: plasma concentration of ibuprofen (0-100 µg/mL)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # Pharmacokinetics parameters
        self.therapeutic_range = (10, 50)  # Target therapeutic range for the drug
        self.half_life = 2.0  # Half-life of the drug in hours
        self.clearance_rate = 0.693 / self.half_life  # Clearance rate calculated using half-life
        self.time_step_hours = 1  # Time step in hours
        self.bioavailability = 0.9  # Proportion of the drug absorbed into systemic circulation
        self.volume_of_distribution = 0.15  # Volume of distribution in L/kg
        self.max_steps = 24  # Maximum simulation steps (24 hours)

        # Initial states
        self.current_step = 0
        self.plasma_concentration = 0.0
        self.normalize = normalize
        self.state_buffer = []  # Stores historical plasma concentrations for normalization

    def reset(self, seed=None, **kwargs):
        """
        Resets the environment to the initial state.

        Args:
            seed (int, optional): Random seed for reproducibility.

        Returns:
            tuple: The initial state and an empty info dictionary.
        """
        super().reset(seed=seed)
        self.current_step = 0  # Reset step counter
        self.plasma_concentration = 0.0  # Reset plasma concentration
        self.state_buffer = []  # Clear state buffer
        state = np.array([self.plasma_concentration], dtype=np.float32)  # Initial state
        return self._normalize(state), {}

    def step(self, action):
        """
        Takes a step in the environment based on the selected action.

        Args:
            action (int): The chosen action, representing the dose (0-4 units).

        Returns:
            tuple: Contains the next state, reward, done flag, truncated flag, and info dictionary.
        """
        # Calculate the administered dose (200 mg per unit of action)
        dose_mg = action * 200
        # Calculate the absorbed dose after bioavailability
        absorbed_mg = dose_mg * self.bioavailability
        # Calculate the resulting plasma concentration (µg/mL)
        absorbed_concentration = absorbed_mg / (self.volume_of_distribution * 70)  # Assume 70 kg body weight
        # Update plasma concentration with the absorbed dose
        self.plasma_concentration += absorbed_concentration
        # Apply clearance (exponential decay)
        self.plasma_concentration *= np.exp(-self.clearance_rate * self.time_step_hours)

        # Generate the new state
        state = np.array([self.plasma_concentration], dtype=np.float32)
        normalized_state = self._normalize(state)  # Normalize the state if enabled

        # Append current concentration to state buffer
        self.state_buffer.append(self.plasma_concentration)

        # Calculate reward based on the therapeutic range
        if self.therapeutic_range[0] <= self.plasma_concentration <= self.therapeutic_range[1]:
            reward = 10  # Positive reward for being in the therapeutic range
        else:
            # Penalize for concentrations outside the therapeutic range
            if self.plasma_concentration < self.therapeutic_range[0]:
                reward = -5 - (self.therapeutic_range[0] - self.plasma_concentration) * 0.5
            elif self.plasma_concentration > self.therapeutic_range[1]:
                reward = -5 - (self.plasma_concentration - self.therapeutic_range[1]) * 0.5

        # Apply a severe penalty for toxic concentrations (above 100 µg/mL)
        if self.plasma_concentration > 100:
            reward -= 15

        # Update step counter and check termination conditions
        self.current_step += 1
        done = self.current_step >= self.max_steps  # Episode ends after max steps
        truncated = False  # No explicit truncation condition in this environment
        info = {}  # Additional diagnostics can be added to this dictionary

        return normalized_state, reward, done, truncated, info

    def _normalize(self, state):
        """
        Normalizes the state based on the mean and standard deviation of past concentrations.

        Args:
            state (numpy.ndarray): The current state (plasma concentration).

        Returns:
            numpy.ndarray: The normalized state if normalization is enabled; otherwise, the original state.
        """
        if self.normalize and len(self.state_buffer) > 1:
            mean = np.mean(self.state_buffer)
            std = np.std(self.state_buffer) + 1e-8  # Add a small value to prevent division by zero
            return (state - mean) / std
        return state
