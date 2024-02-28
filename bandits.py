"""
Bandits Source File

Collection of various Bandit types useful for the k-armed bandit problem.

Requires `numpy` to be installed.

Currently contains implementations for:
    - Stationary Bandit

TODO: Add implementations for:
    - Nonstationary Bandit
"""

import numpy as np

class Bandit:
    def selectAction(self, a):
        """
        Returns the associated reward for a given action

        Parameters
        ----------
        a : str
            Some indicator to select which action to take
        """
        pass


class StationaryBandit(Bandit):
    """
    Python implementation of the stationary k-bandit concept. Extends the "Bandit" interface

    ...

    Attributes
    ----------
    k : int
        number of "arms" (valid actions) the bandit has (default 3)
    actions : np.array
        list of all actions and their rewards where:
            - index = action ID
            - value = cooresponding reward
        ex: [4,2] is an array where
            - Action 0 has a reward of 4
            - Action 1 has a reward of 2
    min : int
        minimum value for the reward (default 0)
    max : int
        maximum value for the reward (default 10)
    variance : int
        normal distribution variance value (default 1)
    
    Methods
    -------
    selectAction(a)
        Returns the associated action value as a standard distribution with:
            - mean = int a
            - variance = 1
    """

    def __init__(self, k: int = 3, min: int = 0, max: int = 10, variance: int = 1) -> None:
        """
        Parameters
        ----------
        k : int
            number of "arms" (valid actions) the bandit has (default 3)
        min : int
            minimum value for the reward (default 0)
        max : int
            maximum value for the reward (default 10)
        variance : int
            normal distribution variance value (default 1)
        """
        #TODO: Add "ValueError" checks for this constructor
        
        self.k = k
        self.min = min
        self.max = max
        self.variance = variance
        self.actions = np.random.randint(min, max, k)

    def selectAction(self, a: int) -> float:
        """
        Returns the associated reward for a given action

        Parameters
        ----------
        a : str
            Which action to take (from 0 to k)

        Raises
        ------
        Value Error
            If selected action is not within the range of accepted "k" actions
        """

        if a not in range(0,self.k):
            raise ValueError("Invalid Action, out of range")
        else:
            return np.random.normal(self.actions[a],self.variance)
        
#class NonstationaryBandit(Bandit):