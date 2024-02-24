import numpy as np

# Simple stationay bandit class
class Bandit:
    # Variables
    k = 0
    actions = np.array([])

    # Constructor
    def __init__(self, k):  
        self.k = k
        self.actions = np.random.randint(0, 100, k) # Rewards always between 0 and 100. Can easily make customizable as well, if desired

    # Function
    def selectAction(self, a):
        if a not in range(0,self.k):
            print("Invalid Action, out of range")
            return -1
        else:
            return np.random.normal(self.actions[a],1) # Return normal distribution with mean equalling randomized reward value and spread equalling 1

# Simple Epsilon Greedy Agent Implementation
class EpsilonGreedyAgent:
    # Variables
    bandit = Bandit(0)
    epsilon = 0 # Chance of choosing any random action rather than greedy action

    q_star = 0
    reward_estimates = np.array([])
    reward_select_counts = np.array([])

    # Constructor
    def __init__(self, bandit, epsilon):
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

print("How many arms would you like the bandit to have? ")
input_K = input()
curr_bandit = Bandit(int(input_K))

print("What would you like your epsilon value for the agent to be? (Decimal Form Only) ")
input_epsilon = input()
curr_agent = EpsilonGreedyAgent(curr_bandit, float(input_epsilon))

print("How many times would you like the agent to be able to choose an action? ")
input_num_runs = input()

print_interval = 100
for i in range(0,int(input_num_runs)):
    curr_agent.chooseAction()
    if i % print_interval == 0:
        print(f"Current Reward Estimate: {curr_agent.reward_estimates}")

print(f"Final Reward Estimate: {curr_agent.reward_estimates}")
print(f"Actual Action Reward Values: {curr_bandit.actions}")