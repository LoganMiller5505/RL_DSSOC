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
print_interval = 10

for i in range(0,n):
    greedy_agent.chooseAction()
    opt_greedy_agent.chooseAction()
    eps_greedy_agent.chooseAction()
    random_agent.chooseAction()
    if i % print_interval == 0:
        print(f"Greedy Reward Estimate at Step #{i}: {greedy_agent.reward_estimates}")
        print(f"Optimistic Greedy Reward Estimate at Step #{i}: {opt_greedy_agent.reward_estimates}")
        print(f"Epsilon Greedy Reward Estimate at Step #{i}: {eps_greedy_agent.reward_estimates}")
        print("-----------------------------------------------")

print(f"Final Greedy Reward Estimate: {greedy_agent.reward_estimates}")
print(f"Final Optimistic Greedy Reward Estimate: {opt_greedy_agent.reward_estimates}")
print(f"Final Epsilon Greedy Reward Estimate: {eps_greedy_agent.reward_estimates}")
print(f"Actual Action Reward Values: {curr_bandit.actions}")
print("-----------------------------------------------")
print(f"Greedy Total Points: {greedy_agent.total_points}")
print(f"Optimistic Greedy Total Points: {opt_greedy_agent.total_points}")
print(f"Epsilon Greedy Total Points: {eps_greedy_agent.total_points}")
print(f"Random Total Points: {random_agent.total_points}")
print("-----------------------------------------------")
print(f"Greedy Avg Points/Step: {greedy_agent.total_points/n}")
print(f"Optimistic Greedy Avg Points/Step: {opt_greedy_agent.total_points/n}")
print(f"Epsilon Greedy Avg Points/Step: {eps_greedy_agent.total_points/n}")
print(f"Random Avg Points/Step: {random_agent.total_points/n}")