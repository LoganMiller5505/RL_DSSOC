import numpy as np
from bandits import StationaryBandit
from agents import EpsilonGreedyAgent

# TODO: Add futher user input, if desired

# Define bandit constants
print("How many arms would you like the bandit to have? ")
k = int(input())
min = 0
max = 10
variance = 1

# Define agent constants
print("What would you like your epsilon value for the agent to be? (Decimal Form Only) ")
epsilon = float(input())

# Create bandit & agent objects
curr_bandit = StationaryBandit(k, min, max, variance)
curr_agent = EpsilonGreedyAgent(curr_bandit, epsilon)

# Define runtime & output constants
print("How many times would you like the agent to be able to choose an action? ")
n = int(input())
print_interval = 10

for i in range(0,n):
    curr_agent.chooseAction()
    if i % print_interval == 0:
        print(f"Current Reward Estimate at Step #{i}: {curr_agent.reward_estimates}")

print(f"Final Reward Estimate: {curr_agent.reward_estimates}")
print(f"Actual Action Reward Values: {curr_bandit.actions}")