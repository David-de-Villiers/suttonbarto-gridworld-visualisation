import numpy as np
import random

class QLearningAgent:
    """
    Q-learning agent with epsilon-greedy exploration.
    """
    def __init__(self, rows=5, cols=5, num_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        """Simple Q-Learning Agent

        :param rows: Number of rows in gridworld, defaults to 5
        :param cols: Number of columns in gridworld, defaults to 5
        :param num_actions: UP/DOWN/LEFT/RIGHT, defaults to 4
        :param alpha: Learning rate of agent, defaults to 0.1
        :param gamma: Discount factor for future rewards, defaults to 0.9
        :param epsilon: Exploration rate of agent, defaults to 0.1
        """
        self.rows = rows
        self.cols = cols
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table to store value of states and actions - shape (5, 5, 4)
        self.Q = np.zeros((rows, cols, num_actions))


    def get_action(self, state):
        """Exploration vs Exploitation

        :param state: x and y position of cell in gridworld
        :return: Chosen action of agent (UP/DOWN/LEFT/RIGHT)
        """
        x, y = state
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.Q[x, y])

    def update(self, state, action, reward, next_state):
        """Update Q-values using the Q-learning update rule.

        :param state: Relevant state to be updated in Q-table
        :param action: Action taken in state
        :param reward: Reward received after action
        :param next_state: State reached after action
        """
        y, x = state
        ny, nx = next_state

        old_val = self.Q[y, x, action]
        next_max = np.max(self.Q[ny, nx])

        new_value = old_val + self.alpha * (reward + self.gamma * next_max - old_val)
        self.Q[y, x, action] = new_value