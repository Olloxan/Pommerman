
import os
import sys
import numpy as np

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility


# Instantiate the environment
config = ffa_v0_fast_env()
env = Pomme(**config["env_kwargs"])

# Add 3 random agents
agents = {}
for agent_id in range(1):
    agents[agent_id] = SimpleAgent(config["agent"](agent_id, config["game_type"]))

# Add human agent
agent_id += 1
agents[3] = PlayerAgent(config["agent"](agent_id, config["game_type"]), "arrows")

env.set_agents(list(agents.values()))
env.set_init_game_state(None)


# Seed and reset the environment
env.seed(0)
obs = env.reset()

#row = [4,4,4,4,5,6,7,8,9,10,11]

first_boardrow = env._board[0]

# Run the agents until we're done
done = False
while not done:
    env.render()
    actions = [5,5]
    
    obs, reward, done, info = env.step(actions)
    #kacka = featurize(obs[0])
env.render(close=True)
env.close()

# Print the result
print(info)

# ----------------------------------
env = Pomme(**config["env_kwargs"])
agents = {}
for agent_id in range(2):
    agent = TrainingAgent(config["agent"](agent_id, config["game_type"]))      
    agents[agent_id] = agent
env.set_agents(list(agents.values()))
env.set_init_game_state(None)
num_actions = env.action_space.n

obs = env.reset()
# ---------------------------------