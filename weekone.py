import numpy as np
from bandits import StationaryBandit
from agents import EpsilonGreedyAgent
from agents import GreedyAgent
from agents import OptimisticGreedyAgent
from agents import RandomAgent

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

greedy_agent = GreedyAgent(curr_bandit)
opt_greedy_agent = OptimisticGreedyAgent(curr_bandit, 20)
eps_greedy_agent = EpsilonGreedyAgent(curr_bandit, epsilon)

random_agent = RandomAgent(curr_bandit)


# Define runtime & output constants
print("How many times would you like the agent to be able to choose an action? ")
n = int(input())

print("-----------------------------------------------------")

greedy_agent.runSeqeuence(n, 1000)
opt_greedy_agent.runSeqeuence(n,1000)
eps_greedy_agent.runSeqeuence(n, 1000)
random_agent.runSeqeuence(n)

print(f"\n\nTRUE ACTION VALUES: {curr_bandit.actions}\n\n")
print("-----------------------------------------------------")

# Test functionality of reset
greedy_agent.reset()
opt_greedy_agent.reset()
eps_greedy_agent.reset()
random_agent.reset()

greedy_agent.runSeqeuence(n, 1000)
opt_greedy_agent.runSeqeuence(n,1000)
eps_greedy_agent.runSeqeuence(n, 1000)
random_agent.runSeqeuence(n)

print(f"\n\nTRUE ACTION VALUES: {curr_bandit.actions}\n\n")
print("-----------------------------------------------------")