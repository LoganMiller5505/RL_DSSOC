import numpy as np
from bandits import StationaryBandit
from agents import EpsilonGreedyAgent
from agents import GreedyAgent
from agents import OptimisticGreedyAgent
from agents import RandomAgent
from agents import UpperConfidenceBoundAgent

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
optimistic_val = 20

c = 1

# Create bandit & agent objects
curr_bandit = StationaryBandit(k, min, max, variance)

greedy_agent = GreedyAgent(curr_bandit)
opt_greedy_agent = OptimisticGreedyAgent(curr_bandit, optimistic_val)
eps_greedy_agent = EpsilonGreedyAgent(curr_bandit, epsilon)
ucb_agent = UpperConfidenceBoundAgent(curr_bandit, c)

random_agent = RandomAgent(curr_bandit)


# Define runtime & output constants
print("How many times would you like the agent to be able to choose an action? ")
n = int(input())
print_frequency = 1000

print("-----------------------------------------------------")

greedy_agent.runSequence(n, print_frequency)
opt_greedy_agent.runSequence(n,print_frequency)
eps_greedy_agent.runSequence(n, print_frequency)
ucb_agent.runSequence(n, print_frequency)
random_agent.runSequence(n)

print(f"\n\nTRUE ACTION VALUES: {curr_bandit.actions}\n\n")
print("-----------------------------------------------------")

# Test functionality of reset
greedy_agent.reset()
opt_greedy_agent.reset()
eps_greedy_agent.reset()
ucb_agent.reset()
random_agent.reset()

greedy_agent.runSequence(n)
opt_greedy_agent.runSequence(n)
eps_greedy_agent.runSequence(n)
ucb_agent.runSequence(n)
random_agent.runSequence(n)

print(f"\n\nTRUE ACTION VALUES: {curr_bandit.actions}\n\n")
print("-----------------------------------------------------")



# Test functionalty of changing bandit
new_bandit = StationaryBandit(k+1, min, max, variance)

greedy_agent.changeBandit(new_bandit)
opt_greedy_agent.changeBandit(new_bandit)
eps_greedy_agent.changeBandit(new_bandit)
ucb_agent.changeBandit(new_bandit)
random_agent.changeBandit(new_bandit)

greedy_agent.runSequence(n)
opt_greedy_agent.runSequence(n)
eps_greedy_agent.runSequence(n)
ucb_agent.runSequence(n)
random_agent.runSequence(n)

print(f"\n\nTRUE ACTION VALUES: {new_bandit.actions}\n\n")
print("-----------------------------------------------------")

# Test functionality of reset
greedy_agent.reset()
opt_greedy_agent.reset()
eps_greedy_agent.reset()
ucb_agent.reset()
random_agent.reset()

greedy_agent.runSequence(n)
opt_greedy_agent.runSequence(n)
eps_greedy_agent.runSequence(n)
ucb_agent.runSequence(n)
random_agent.runSequence(n)

print(f"\n\nTRUE ACTION VALUES: {new_bandit.actions}\n\n")
print("-----------------------------------------------------")