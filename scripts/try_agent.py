import sys
from drlqap.agent_configs import agents
from drlqap.taskgenerators import generators

agent = agents[sys.argv[1]]()
qap = generators["small_fixed"].sample()
print(sum(p.numel() for p in agent.policy_net.parameters()))
print(agent.solve_and_learn(qap))
