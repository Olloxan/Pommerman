import os
import sys
import numpy as np

from pommerman.agents import SimpleAgent, TrainingAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

from pommerman.constants import GameType
# Instantiate the environment
config = ffa_v0_fast_env()
#config['game_type'] = GameType.OneVsOne
#config['env_kwargs']['game_type'] = GameType.OneVsOne

#FFA = 1
#   Team = 2
#   TeamRadio = 3
#   OneVsOne = 4

#env_kwargs = {
#        'game_type': game_type,
#        'board_size': 11,
#        'num_rigid': 36,
#        'num_wood': 36,
#        'num_items': 20,
#        'first_collapse': constants.FIRST_COLLAPSE,
#        'max_steps': 800,
#        'render_fps': 1000,
#        'agent_view_size': constants.AGENT_VIEW_SIZE,
#        'is_partially_observable': True,
#        'env': env_entry_point,
#    }

env = Pomme(**config["env_kwargs"])

# actions

# actions
#nothing : 0
#key.UP : 1,
#key.DOWN : 2,
#key.LEFT : 3,
#key.RIGHT : 4,
#key.SPACE : 5, --> Bomb

# Add four random agents
agents = {}

agents[0] = SimpleAgent(config["agent"](0, config["game_type"]))
agents[1] = SimpleAgent(config["agent"](1, config["game_type"]))
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
    actions[0] = 0
    obs, reward, done, info = env.step(actions)
env.render(close=True)
env.close()

print(info)
