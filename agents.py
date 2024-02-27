import numpy as np
from bandits import Bandit

# Simple Epsilon Greedy Agent Implementation
class EpsilonGreedyAgent:
    """
    Python implementation of an Epsilon Greedy Agent

    ...

    Attributes
    ----------
    
    
    Methods
    -------

    """

    # Variables
    bandit = Bandit(0)
    epsilon = 0 # Chance of choosing any random action rather than greedy action

    q_star = 0
    reward_estimates = np.array([])
    reward_select_counts = np.array([])

    # Constructor
    def __init__(self, bandit: Bandit, epsilon: float):
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Invalid Epsilon, must be within (0,1)")
        self.bandit = bandit
        self.epsilon = epsilon
        self.reward_estimates.resize(bandit.k)
        self.reward_select_counts.resize(bandit.k)

    # Functions
    def updateRewards(self, selected_action, selected_reward):
        # Using derived formulas from RL textbook
        q = self.reward_estimates[selected_action]
        n = self.reward_select_counts[selected_action]
        if(n==0): #Action has NOT been selected before
            self.reward_estimates[selected_action] = selected_reward
        else:
            self.reward_estimates[selected_action] = q + ( (1/n) * (selected_reward - q) )
        self.reward_select_counts[selected_action] += 1

    def chooseAction(self):
        k = self.bandit.k
        selected_action = 0
        # Random value [0,1) to determine if random action will be used rather than greedy
        epsilon_check = np.random.random()

        print_substr = "GREEDILY"

        if(self.epsilon > epsilon_check): # Random action
            selected_action = np.random.randint(0,k)
            print_substr = "EPSILON-RANDOMLY"
        else: # Greedy action
            selected_action = self.reward_estimates.argmax()

        selected_reward = self.bandit.selectAction(selected_action) # Reward of selected action through bandit
        self.updateRewards(selected_action, selected_reward)

#class OptimisticGreedyAgent:

#class GreedyAgent:

#class RandomAgent: