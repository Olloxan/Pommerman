import os
import sys
import numpy as np

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32) # array
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32) # array
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32) # array
    position = make_np_float(obs["position"]) # coord
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]]) # Zahl fuer jeden Player
    can_kick = make_np_float([obs["can_kick"]]) # bool

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))

# Instantiate the environment
config = ffa_v0_fast_env()
env = Pomme(**config["env_kwargs"])
env.action_space.n
# Add 3 random agents
agents = {}
for agent_id in range(3):
    agents[agent_id] = SimpleAgent(config["agent"](agent_id, config["game_type"]))

# Add human agent

agent_id += 1
agents[3] = PlayerAgent(config["agent"](agent_id, config["game_type"]), "arrows")

env.set_agents(list(agents.values()))
env.set_init_game_state(None)


# Seed and reset the environment
env.seed(0)
obs = env.reset()

# Run the agents until we're done
done = False
while not done:
    env.render()
    actions = env.act(obs) # brauch ich nicht
    #actions = [action % 4 for action in actions]
    #actions = [0,actions[1]]
    obs, reward, done, info = env.step(actions)
    #kacka = featurize(obs[0])
env.render(close=True)
env.close()

# Print the result
print(info)