import numpy as np

class GridworldEnv:
    """
    5x5 Gridworld with special states A, B and transitions to A', B'.
    Setup according to Sutton & Barto Example 3.5.
    """
    def __init__(self, discount=0.9):
        self.rows = 5
        self.cols = 5
        self.discount = discount

        # Special states (row, col)
        self.A = (0, 1)
        self.A_prime = (4, 1)
        self.B = (0, 3)
        self.B_prime = (2, 3)

        # Actions - UP,DOWN,LEFT,RIGHT
        self.action_space = [0, 1, 2, 3]
        self.num_actions = 4

        # Agent state
        self.state = None


    def reset(self):
        """Reset agent to a random cell

        :return: New cell of agent
        """
        row = np.random.randint(self.rows)
        col = np.random.randint(self.cols)
        self.state = (row, col)
        return self.state


    def step(self, action):
        """
        Select action and return new state, reward and information
        NB Done condition is always false here

        :param action: 0=Up, 1=Down, 2=Left, 3=Right
        :return: next_state, reward, done, info
        """
        y, x = self.state

        # Check if in special states A or B
        if self.state == self.A:
            # +10 then goes to A'
            self.state = self.A_prime
            return self.state, 10.0, False, {}
        if self.state == self.B:
            # +5 then goes to B'
            self.state = self.B_prime
            return self.state, 5.0, False, {}

        # Otherwise, normal transitions for UP,DOWN,LEFT,RIGHT
        if action == 0:
            ny, nx = y-1, x
        elif action == 1:
            ny, nx = y+1, x
        elif action == 2:
            ny, nx = y, x-1
        else:
            ny, nx = y, x+1

        # If out of bounds, stay in place, reward of -1
        if not (0 <= ny < self.rows and 0 <= nx < self.cols):
            return self.state, -1.0, False, {}

        # Otherwise move and reward 0
        self.state = (ny, nx)
        return self.state, 0.0, False, {}
