"""This file implements the `SmallGrid` and `MediumGrid` examples by
defining the grid size, obstacles, and win/lose states of the grid.

See `gridworld.py` for the implementation of the base GridWorld class.

@author: Scott Brown
"""

from gridworld import GridWorld


# This syntax define `SmallGrid` to be a "subclass" of `GridWorld`,
# meaning that it inherits all the methods we defined in gridworld.py
class SmallGrid(GridWorld):
    n = 6  # size in x dimension
    m = 5  # size in y dimension

    obstacles = [
        [0, 5, 0, 0], [0, 0, 0, 4], [0, 5, 4, 4],
        [5, 5, 0, 4], [2, 2, 2, 2],
    ]

    start = (1, 1)

    win_states = {(4, 3)}   # States that give positive reward
    lose_states = {(4, 2)}  # States that give negative reward


class MediumGrid(GridWorld):
    n = 7  # size in x dimension
    m = 7  # size in y dimension

    obstacles = [
        [0, 6, 0, 0], [0, 0, 0, 6], [0, 6, 6, 6],
        [6, 6, 0, 6], [2, 2, 3, 4], [4, 4, 3, 3],
    ]

    start = (1, 2)

    # The medium grid has several win/lose states, which may have
    # different values depending on the cost function being used
    distant_exit = (5, 3)
    close_exit = (3, 3)

    big_lose_states = {(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)}
    small_lose_states = {(4, 2)}

    win_states = {distant_exit, close_exit}
    lose_states = big_lose_states | small_lose_states


class MediumGridBridge(MediumGrid):
    """This class is the same as MediumGrid, but overrides the cost
    function to be the 'bridge' version described in the homework.

    """

    # When subclasses define a method with the same name as one in the
    # parent class, that method is replaced (called "overriding"). So
    # when `cost` is called on a `MediumGridBridge` object, this
    # function will be used
    def cost(self, x):
        """Implements the 'bridge' cost from part (d)"""
        if x == self.close_exit:  # Small win state
            return -1
        elif x == self.distant_exit:  # Big win state
            return -10
        elif x in self.small_lose_states:
            return 1
        elif x in self.big_lose_states:
            return 10
        else:
            return 0
