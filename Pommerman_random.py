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


# Add four random agents
agents = {}
for agent_id in range(4):
    agents[agent_id] = RandomAgent(config["agent"](agent_id, config["game_type"]))
env.set_agents(list(agents.values()))
env.set_init_game_state(None)

# Seed and reset the environment
env.seed(0)
obs = env.reset()

# Run the random agents until we're done
done = False
while not done:
    env.render()
    actions = env.act(obs)
    obs, reward, done, info = env.step(actions)
env.render(close=True)
env.close()

print(info)