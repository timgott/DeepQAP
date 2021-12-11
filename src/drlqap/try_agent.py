import sys
from drlqap.agent_configs import agents
from drlqap.taskgenerators import small_random_graphs

agent = agents[sys.argv[1]]()
qap = small_random_graphs()
print(agent.solve_and_learn(qap))