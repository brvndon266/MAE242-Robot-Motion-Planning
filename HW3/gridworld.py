"""This file implements all of the functions required for creating,
manipulating, and visualizing 2D grid environments.

@author: Tor Anderson and Scott Brown

"""

import numpy as np
import matplotlib.pyplot as plt


class GridWorld():
    """This class holds the data and functions for a generic grid
    world implementation.

    This class represents an empty grid. It is meant to be inherited
    by other classes (such as SmallGrid and MediumGrid) which will
    define the grid size and obstacles for that specific case.  The
    purpose of this class is to implement functions that are useful
    and common to all possible grid worlds.

    """
    n = 0           # size in x dimension
    m = 0           # size in y dimension
    obstacles = []  # list of obstacles in the grid

    start = (0, 0)       # Starting node
    win_states = set()   # States that give positive reward
    lose_states = set()  # States that give negative reward

    u_set = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Set of valid inputs
    n_a = len(u_set)                            # Number of input options

    # The __init__ function is a special method in a Python class,
    # which is called whenever an object of that class is created. In
    # this case, we use it to set up the basic data associated with
    # the grid, which is parsed from the list of obstacles.
    def __init__(self):
        """Initialize all the data for the grid world."""

        # Build the grid. 1 represents free space, 0 represent obstacle
        self.grid = np.ones((self.n, self.m))

        for obs in self.obstacles:
            west, east, south, north = obs
            for i in range(west, east+1):
                for j in range(south, north+1):
                    self.grid[i][j] = 0

        # Create a list of all valid states
        self.states = [
            (i, j) for i in range(self.n) for j in range(self.m)
            if self.grid[i][j]
        ]

        # Create image for plotting the grid. Free space is white,
        # obstacles are black. Image has dims (x, y, rgb)
        self.img = np.repeat(self.grid[:, :, np.newaxis], 3, axis=2)
        self.img[self.start] = [0, 0, 1]  # Start node plotted in blue
        for s in self.win_states:    # Winning nodes plotted in green
            self.img[s] = [0, 1, 0]
        for s in self.lose_states:   # Losing nodes plotted in red
            self.img[s] = [1, 0, 0]

    def cost(self, x):
        """Return the cost of state `x`"""
        if x in self.lose_states:
            return 1
        elif x in self.win_states:
            return -1
        else:
            return 0

    def validateInput(self, u):
        """Check that an input `u` is in the valid set"""
        if u not in self.u_set:
            raise ValueError(f"Invalid input u: {u}")

    def nextState(self, x, u):
        """Calculate the state following `x` under action `u`.

        Returns `x` + `u` unless that would result in a collision with
        an obstacle, in which case it returns `x`.

        """

        self.validateInput(u)
        x_p = tuple(x[i] + u[i] for i in range(len(x)))

        return x_p if self.grid[x_p] else x

    def nextStateRandom(self, x, u, eta):
        """Return the state following `x` under action `u`

        There is a (1-`eta`) chance that the input is corrupted, in
        which case there is a 50/50 chance of taking the action either
        left or right of the chosen `u`.

        """
        self.validateInput(u)

        p = np.random.random_sample()
        if p > 1 - eta:
            # Introduce a movement error
            # Rotate the direction of the input +/- 90 degrees
            # Randomly choose positive or negative half the time
            sign = 1 if 0 <= p <= eta/2 else -1
            u = ((0, sign) if u[0] else (sign, 0))

        return self.nextState(x, u)

    def getNominalPathFromPolicy(self, pi):
        """Compute the nominal (i.e., noise-free) path from the start
        to the goal under policy `pi`. If the goal is not reached in
        20 steps, prints a warning and returns.

        """

        MAX_LENGTH = 20
        x = self.start

        path = []
        while x not in self.win_states:
            u = self.u_set[pi[x]]
            x = self.nextState(x, u)
            path.append(x)

            if len(path) > MAX_LENGTH:
                print("Warning: path exceeded max length. The policy may have \
an infinite loop, or does not reach the goal.")
                break

        return path

    def getRandomPathFromPolicy(self, pi, eta):
        """Compute a noise-corrupted path from the start to the goal
        under policy `pi`. If the goal is not reached in 50 steps,
        prints a warning and returns.

        """
        MAX_LENGTH = 50
        x = self.start

        path = []
        while x not in self.win_states:
            u = self.u_set[pi[x]]
            x = self.nextStateRandom(x, u, eta)
            path.append(x)

            if len(path) > MAX_LENGTH:
                print("Warning: path exceeded max length. The policy may have \
an infinite loop, or does not reach the goal.")
                break

        return path

    ###################################
    # Plotting and printing functions #
    ###################################

    def plot(self, img=None, show=False):
        """Plot the maze.

        Can also plot an overlay of, for example, the value function
        if the `img` argument is used.

        `show` controls whether plt.show() is called

        """
        if img is None:
            img = self.img.copy()

        fig, ax = plt.subplots()
        ax.imshow(img.swapaxes(0, 1))
        ax.invert_yaxis()

        if show:
            plt.show()

    def plotValues(self, V, title="", show=False):
        """Plot the value function (matrix `V`) over the grid.
        Red is low value, green is high.

        """
        v_min = np.min(V)
        v_max = np.max(V)

        img = self.img.copy()
        for s in self.states:
            v_rel = (V[s] - v_min)/(v_max - v_min)
            img[s] = [1-v_rel, v_rel, 0]

        self.plot(img, show)

    def plotPath(self, path, title="", show=False):
        """Plot `path`, a list of grid coordinates. The color fades
        from blue to green over the length of the path.

        """
        img = self.img.copy()
        N = len(path)
        i = 0.0
        for x in path:
            i += 1
            img[x] = [0, i/N, 1 - i/N]  # blue -> green

        self.plot(img, show)

    def print(self, labels):
        """Print the grid with `labels` in each cell"""
        line = "--------" * self.n + "-"
        for j in reversed(range(0, self.m)):
            print(line)
            row = "| "
            for i in range(0, self.n):
                row += str(labels[i, j])[:5] if self.grid[i, j] else " Obs "
                row += " | "
            print(row)
        print(line)

    def printValues(self, V):
        """Print the value function to the terminal"""
        print(" Value ".center(self.n*8+1, "+"))
        self.print(labels=V)

    def printPolicy(self, pi):
        """Print the policy (in symbols) to the terminal"""
        u_sym = [" --> ", "  ^  ", "  <--  ", "  v  "]
        print(" Policy ".center(self.n*8, "+"))
        self.print(np.array([[u_sym[a] for a in row] for row in pi]))
