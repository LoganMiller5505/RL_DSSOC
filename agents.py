"""
Agents Source File

Collection of various Agent types useful for the k-armed bandit problem.

Requires `numpy` to be installed, and the bandits source file to be imported correctly.

Currently contains implementations for:
    - Epsilon Greedy Agent
    - Greedy Agent
    - Optimistic Greedy Agent
    - Random Agent
"""

import numpy as np
from bandits import Bandit










class EpsilonGreedyAgent:
    """
    Python implementation of an Epsilon Greedy Agent

    ...

    Attributes
    ----------
    bandit : Bandit
        Associated bandit for the agent to operate on
    total_points : float
        Keeps a running total of all points across all n times "chooseAction" has been called
    epsilon : float
        Chance for model to pick a random (non-greedy) action. Must be between 0 and 1! Reasoning:
            - Epsilon of 0 = Greedy Model
            - Epsilon of 1 = Purely Random Model
        (default 0.1)
    __reward_estimates : np.array
        Estimated value of each action's reward based on prior experience, where:
            - index = action ID
            - value = estimated cooresponding reward
    __reward_select_counts : np.array
        Array that keeps track of how many times each action has been selected, where:
            - index = action ID
            - value = num times action has been selected
    
    Methods
    -------
    chooseAction()
        Uses Epsilon-Greedy logic to select an action and realize its associated reward.
        Passes this information into the (private) updateRewards function.
    runSequence(n = 1000, print_interval = 100)
        Run the model input n amount of times, providing print statements to indicate how it is performing
    """

    total_points = 0

    def __init__(self, bandit: Bandit, epsilon: float = 0.1) -> None:
        """
        Parameters
        ----------
        bandit : Bandit
            Associated bandit for the agent to operate on
        epsilon : float
            Chance for model to pick a random (non-greedy) action. Must be between 0 and 1! Reasoning:
                - Epsilon of 0 = Greedy Model
                - Epsilon of 1 = Purely Random Model
            (default 0.1)
        """
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Invalid Epsilon, must be within (0,1)")
        self.bandit = bandit
        self.epsilon = epsilon
        # TODO: See if there is a better way to do this, maybe in one function. I attempted this, but it lead to very loose, inaccurate floating point problems with the arrays
        # However, for now, this provides desired behavior without issue.
        self.__reward_estimates = np.array([])
        self.__reward_select_counts = np.array([])
        self.__reward_estimates.resize(bandit.k)
        self.__reward_select_counts.resize(bandit.k)

    def __updateRewards(self, selected_action: int, selected_reward: float) -> None:
        """
        Private method which updates "reward_estimates" and "reward_select_counts" based on the input action, reward pair representing what action the model chose and what reward it was provided.
        Uses "Q" value updating formula described in textbook.

        Parameters
        ----------
        selected_action : int
            Action ID component of selected action ID/cooresponding reward pair
        selected_reward : float
            Reward component of selected action ID/cooresponding reward pair
        """
        # Using derived formulas from RL textbook
        q = self.__reward_estimates[selected_action]
        n = self.__reward_select_counts[selected_action]
        if(n==0): #Action has NOT been selected before
            self.__reward_estimates[selected_action] = selected_reward
        else:
            self.__reward_estimates[selected_action] = q + ( (1/n) * (selected_reward - q) )
        self.__reward_select_counts[selected_action] += 1

    def chooseAction(self) -> None:
        """
        Public method which uses Epsilon-Greedy logic to select an action and realize its associated reward.
        Passes this information into the updateRewards function.
        """
        k = self.bandit.k
        selected_action = 0
        # Random value [0,1) to determine if random action will be used rather than greedy
        epsilon_check = np.random.random()

        if(self.epsilon > epsilon_check): # Random action
            selected_action = np.random.randint(0,k)
        else: # Greedy action
            selected_action = self.__reward_estimates.argmax()

        selected_reward = self.bandit.selectAction(selected_action) # Reward of selected action through bandit
        self.total_points += selected_reward
        self.__updateRewards(selected_action, selected_reward)
    
    def runSeqeuence(self, n: int = 1000, print_interval: int = 100) -> None:
        """
        Run the model input n amount of times, providing print statements to indicate how it is performing

        Parameters
        ----------
        n : int
            Number of times to call "chooseAction()" (dfault 1000)
        print_interval : float
            Interval between which to print current reward estimate (default 100)
        """
        for i in range(1,n+1):
            self.chooseAction()
            if i % print_interval == 0:
                print(f"Epsilon Greedy Reward Estimate at Step #{i}: {self.__reward_estimates}")
        print(f"Total Epsilon Greedy Points: {self.total_points}")
        print("-----------------------------------------------------")

    #TODO: Potentially add additional function to reset estimated values so that agent could be "reused" for different problems
        #(could potentialy be used by above function)










class OptimisticGreedyAgent:
    """
    Python implementation of an Optimistic Greedy Agent

    ...

    Attributes
    ----------
    bandit : Bandit
        Associated bandit for the agent to operate on
    total_points : float
        Keeps a running total of all points across all n times "chooseAction" has been called
    __reward_estimates : np.array
        Estimated value of each action's reward based on prior experience, where:
            - index = action ID
            - value = estimated cooresponding reward
    __reward_select_counts : np.array
        Array that keeps track of how many times each action has been selected, where:
            - index = action ID
            - value = num times action has been selected
    
    Methods
    -------
    chooseAction()
        Uses Epsilon-Greedy logic to select an action and realize its associated reward.
        Passes this information into the (private) updateRewards function.
    runSequence(n = 1000, print_interval = 100)
        Run the model input n amount of times, providing print statements to indicate how it is performing
    """

    total_points = 0

    def __init__(self, bandit: Bandit, optimistic_val: float = 50) -> None:
        """
        Parameters
        ----------
        bandit : Bandit
            Associated bandit for the agent to operate on
        optimistic_val : float
            "Optimistic" value to input into model (default 50)
            (NOTE: REQUIRES SOME LEVEL OF KNOWLEDGE ON BANDIT TO PROPERLY SET OPTIMISTIC_VAL since it, generally, must be bigger than the maximum possible reward's mean)
        """
        self.bandit = bandit
        # TODO: See if there is a better way to do this, maybe in one function. I attempted this, but it lead to very loose, inaccurate floating point problems with the arrays
        # However, for now, this provides desired behavior without issue.
        self.__reward_estimates = np.array([])
        self.__reward_select_counts = np.array([])
        self.__reward_estimates.resize(bandit.k)
        self.__reward_select_counts.resize(bandit.k)
        self.__reward_estimates.fill(optimistic_val)

    def __updateRewards(self, selected_action: int, selected_reward: float) -> None:
        """
        Private method which updates "reward_estimates" and "reward_select_counts" based on the input action, reward pair representing what action the model chose and what reward it was provided.
        Uses "Q" value updating formula described in textbook.

        Parameters
        ----------
        selected_action : int
            Action ID component of selected action ID/cooresponding reward pair
        selected_reward : float
            Reward component of selected action ID/cooresponding reward pair
        """
        # Using derived formulas from RL textbook
        q = self.__reward_estimates[selected_action]
        n = self.__reward_select_counts[selected_action]
        if(n==0): #Action has NOT been selected before
            self.__reward_estimates[selected_action] = selected_reward
        else:
            self.__reward_estimates[selected_action] = q + ( (1/n) * (selected_reward - q) )
        self.__reward_select_counts[selected_action] += 1

    def chooseAction(self) -> None:
        """
        Public method which uses Greedy logic to select an action and realize its associated reward.
        Passes this information into the updateRewards function.
        """
        selected_action = self.__reward_estimates.argmax()

        selected_reward = self.bandit.selectAction(selected_action) # Reward of selected action through bandit
        self.total_points += selected_reward
        self.__updateRewards(selected_action, selected_reward)
    
    def runSeqeuence(self, n: int = 1000, print_interval: int = 100) -> None:
        """
        Run the model input n amount of times, providing print statements to indicate how it is performing

        Parameters
        ----------
        n : int
            Number of times to call "chooseAction()" (dfault 1000)
        print_interval : float
            Interval between which to print current reward estimate (default 100)
        """
        for i in range(1,n+1):
            self.chooseAction()
            if i % print_interval == 0:
                print(f"Optimistic Greedy Reward Estimate at Step #{i}: {self.__reward_estimates}")
        print(f"Total Optimistic Greedy Points: {self.total_points}")
        print("-----------------------------------------------------")

    #TODO: Potentially add additional function to reset estimated values so that agent could be "reused" for different problems
        #(could potentialy be used by above function)










class GreedyAgent:
    """
    Python implementation of a Greedy Agent

    ...

    Attributes
    ----------
    bandit : Bandit
        Associated bandit for the agent to operate on
    total_points : float
        Keeps a running total of all points across all n times "chooseAction" has been called
    __reward_estimates : np.array
        Estimated value of each action's reward based on prior experience, where:
            - index = action ID
            - value = estimated cooresponding reward
    __reward_select_counts : np.array
        Array that keeps track of how many times each action has been selected, where:
            - index = action ID
            - value = num times action has been selected
    
    Methods
    -------
    chooseAction()
        Uses Greedy logic to select an action and realize its associated reward.
        Passes this information into the (private) updateRewards function.
    runSequence(n = 1000, print_interval = 100)
        Run the model input n amount of times, providing print statements to indicate how it is performing
    """

    total_points = 0

    def __init__(self, bandit: Bandit) -> None:
        """
        Parameters
        ----------
        bandit : Bandit
            Associated bandit for the agent to operate on
        """
        self.bandit = bandit
        # TODO: See if there is a better way to do this, maybe in one function. I attempted this, but it lead to very loose, inaccurate floating point problems with the arrays
        # However, for now, this provides desired behavior without issue.
        self.__reward_estimates = np.array([])
        self.__reward_select_counts = np.array([])
        self.__reward_estimates.resize(bandit.k)
        self.__reward_select_counts.resize(bandit.k)

    def __updateRewards(self, selected_action: int, selected_reward: float) -> None:
        """
        Private method which updates "reward_estimates" and "reward_select_counts" based on the input action, reward pair representing what action the model chose and what reward it was provided.
        Uses "Q" value updating formula described in textbook.

        Parameters
        ----------
        selected_action : int
            Action ID component of selected action ID/cooresponding reward pair
        selected_reward : float
            Reward component of selected action ID/cooresponding reward pair
        """
        # Using derived formulas from RL textbook
        q = self.__reward_estimates[selected_action]
        n = self.__reward_select_counts[selected_action]
        if(n==0): #Action has NOT been selected before
            self.__reward_estimates[selected_action] = selected_reward
        else:
            self.__reward_estimates[selected_action] = q + ( (1/n) * (selected_reward - q) )
        self.__reward_select_counts[selected_action] += 1

    def chooseAction(self) -> None:
        """
        Public method which uses Greedy logic to select an action and realize its associated reward.
        Passes this information into the updateRewards function.
        """
        selected_action = self.__reward_estimates.argmax()

        selected_reward = self.bandit.selectAction(selected_action) # Reward of selected action through bandit
        self.total_points += selected_reward
        self.__updateRewards(selected_action, selected_reward)
    
    def runSeqeuence(self, n: int = 1000, print_interval: int = 100) -> None:
        """
        Run the model input n amount of times, providing print statements to indicate how it is performing

        Parameters
        ----------
        n : int
            Number of times to call "chooseAction()" (dfault 1000)
        print_interval : float
            Interval between which to print current reward estimate (default 100)
        """
        for i in range(1,n+1):
            self.chooseAction()
            if i % print_interval == 0:
                print(f"Greedy Reward Estimate at Step #{i}: {self.__reward_estimates}")
        print(f"Total Greedy Points: {self.total_points}")
        print("-----------------------------------------------------")

    #TODO: Potentially add additional function to reset estimated values so that agent could be "reused" for different problems
        #(could potentialy be used by above function)










class RandomAgent:
    """
    Python implementation of a Random Agent

    ...

    Attributes
    ----------
    bandit : Bandit
        Associated bandit for the agent to operate on
    total_points : float
        Keeps a running total of all points across all n times "chooseAction" has been called
    
    Methods
    -------
    chooseAction()
        Randomly select an action and realize its associated reward.
    runSequence(n = 1000)
        Run the model input n amount of times, providing a final print statement to indicate how it performed
    """

    total_points = 0

    def __init__(self, bandit: Bandit) -> None:
        """
        Parameters
        ----------
        bandit : Bandit
            Associated bandit for the agent to operate on
        """
        self.bandit = bandit

    def chooseAction(self) -> None:
        """
        Public method which randomly selects an action and realizes its associated reward.
        """
        k = self.bandit.k
        selected_action = 0

        selected_action = np.random.randint(0,k)

        selected_reward = self.bandit.selectAction(selected_action) # Reward of selected action through bandit
        self.total_points += selected_reward

    def runSeqeuence(self, n: int = 1000) -> None:
        """
        Run the model input n amount of times, providing a final print statement to indicate how it performed

        Parameters
        ----------
        n : int
            Number of times to call "chooseAction()" (dfault 1000)
        """
        for i in range(1,n+1):
            self.chooseAction()
        print(f"Total Random Points: {self.total_points}")
        print("-----------------------------------------------------")
    
    #TODO: Potentially add additional function to reset total points so that agent could be "reused" for different problems
        #(could potentialy be used by above function)