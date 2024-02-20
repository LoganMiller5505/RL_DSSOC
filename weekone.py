import numpy as np

class Bandit:
    # Variables
    k = 0
    actions = []

    # Constructor
    def __init__(self, k):  
        self.k = k
        self.actions = np.random.randint(0, 10, k)

    # Function
    def selectAction(self, a):
        if a not in range(0,self.k):
            print("Invalid Action, out of range")
            return -1
        else:
            return np.random.normal(self.actions[a],1)

curr_bandit = Bandit(5)

for i in curr_bandit.actions:
    print(i)
print("")
for _ in range(0,curr_bandit.k):
    print(curr_bandit.selectAction(0))