import os
import sys
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent, TrainingAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

from common.actor_critic import ActorCritic_Large, RolloutStorage
from common.multiprocessing_env import SubprocVecEnv
from common.logger import Logger
from common.myTimer import myTimer

sys.argv.append("AC_Pommer_Large_vs_simple_64x10e6x5_orig")

logger = Logger()
timer = myTimer()

USE_CUDA = torch.cuda.is_available()

# Instantiate the environment
config = ffa_v0_fast_env()

num_envs = 64
num_players = 1

player0 = 0
player1 = 1
player2 = 2
player3 = 3

def make_cuda(input):
    if USE_CUDA:
        return input.cuda()
    return input

# make synchonized parallel environments
def make_env():   
    assert(num_players <= 3)
    def _thunk():
        env = Pomme(**config["env_kwargs"])
        agents = {}
        for agent_id in range(num_players):
            agent = TrainingAgent(config["agent"](agent_id, config["game_type"]))              
            agents[agent_id] = agent
        simple_Agent_id = num_players
        agents[simple_Agent_id] = SimpleAgent(config["agent"](simple_Agent_id, config["game_type"]))
        env.set_agents(list(agents.values()))
        env.set_init_game_state(None)
        return env
    return _thunk

# extract features from observations
def featurize(observations):
    boards=[]
    for observation in observations:        
        boards.append(featureize_single(observation[0]))
    return np.stack(boards)    

def featureize_single(observation):    
    board = np.zeros(state_shape)
    board[0] = observation['board']
    board[1] = observation['bomb_life']        
    board[2] = observation['bomb_blast_strength']    
    return board

def makeTrainingObservation():
    env = Pomme(**config["env_kwargs"])
    agents = {}
    for agent_id in range(num_players):
        agent = TrainingAgent(config["agent"](agent_id, config["game_type"]))              
        agents[agent_id] = agent
    simple_Agent_id = num_players
    agents[simple_Agent_id] = SimpleAgent(config["agent"](simple_Agent_id, config["game_type"]))
    env.set_agents(list(agents.values()))
    env.set_init_game_state(None)
    return env

if __name__ == '__main__':
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)    
    
    state_shape = (3,11,11) 
    num_actions = envs.action_space.n

     #a2c hyperparams:
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    num_steps = 5
    num_frames = int(10e6)

    #rmsprop hyperparams:
    lr    = 7e-4
    eps   = 1e-5
    alpha = 0.99

    #Init a2c and rmsprop
    agent1 = ActorCritic_Large(state_shape, num_actions)
    
    optimizer1 = optim.RMSprop(agent1.parameters(), lr, eps=eps, alpha=alpha)    

    agent1 = make_cuda(agent1)    

    rollout1 = RolloutStorage(num_steps, num_envs, state_shape)   

    if USE_CUDA:        
        rollout1.cuda()       

    all_rewards1 = []
    all_losses1  = []
    
    observations = envs.reset() 
    next_observation = observations
    state = featurize(observations)
    state = make_cuda(torch.FloatTensor(state))

    rollout1.states[0].copy_(state)    

    episode_rewards1 = torch.zeros(num_envs, 1)
    final_rewards1   = torch.zeros(num_envs, 1)    
   
    timer.update(time.time())
    swich_variable = 0

    ##### training observation ######
    traiobsenv = makeTrainingObservation()
    trainobs = traiobsenv.reset()
    #################################

    for i_update in range(num_frames):
        for step in range(num_steps):                             
            
            # actor1 acts in all parallel envs
            action_p1 = agent1.act(make_cuda(state)).squeeze(1).cpu().numpy()                        
            actions_simple = envs.act(next_observation[:,1])
            # separate actions
            action_tuples = []              
            for i in range(num_envs):
                actions = []
                actions.append(action_p1[i])    # player1                
                actions.append(actions_simple[i,1])
                action_tuples.append(actions)
            
            next_observation, reward, finished, _ = envs.step(action_tuples)    # pass actions to environments

            # separate rewards
            reward1 = []                    
           
            for i in range(num_envs):
                reward1.append(reward[i][player0]) # player1           

            reward1 = torch.FloatTensor(reward1).unsqueeze(1) # player1
            
            episode_rewards1 += reward1 # player1            

            finished_masks = torch.FloatTensor(1-np.array(finished)).unsqueeze(1)                                                       

            # final rewards player1
            final_rewards1 *= finished_masks
            final_rewards1 += (1-finished_masks) * episode_rewards1                                              

            episode_rewards1 *= finished_masks # player1            
                                                                       
            finished_masks = make_cuda(finished_masks)

            state = make_cuda(torch.FloatTensor(np.float32(featurize(next_observation))))
            rollout1.insert(step, state, torch.FloatTensor(action_p1).unsqueeze(1), reward1, finished_masks) # player1            

        # v(s_t+1) player1
        _, next_value_p1 = agent1(rollout1.states[-1])
        next_value_p1 = next_value_p1.data        

        # n-step returns player1
        returns_p1 = rollout1.compute_returns(next_value_p1, gamma)        

        # eval actions player1
        logit_p1, action_log_probs_p1, values_p1, entropy_p1 = agent1.evaluate_actions(
            rollout1.states[:-1].view(-1, *state_shape),
            rollout1.actions.view(-1, 1)
        )       

        # advantages player1
        values_p1 = values_p1.view(num_steps, num_envs, 1)
        action_log_probs_p1 = action_log_probs_p1.view(num_steps, num_envs, 1)        
        advantages_p1 = returns_p1 - values_p1

        value_loss_p1 = advantages_p1.pow(2).mean()        
        action_loss_p1 = -(advantages_p1.data * action_log_probs_p1).mean()       

        # optimize player1
        optimizer1.zero_grad()        
        loss_p1 = value_loss_p1 * value_loss_coef + action_loss_p1 - entropy_p1 * entropy_coef
        
        loss_p1.backward()
        nn.utils.clip_grad_norm_(agent1.parameters(), max_grad_norm)
        optimizer1.step()               

        if i_update % 5 == 0:            
             # training observation #
            traiobsenv.render()        
            
            train_actions = traiobsenv.act(trainobs)

            trainobs_state = make_cuda(torch.FloatTensor(np.float32(featureize_single(trainobs[0]))))
            training_action_p1 = agent1.act(trainobs_state.unsqueeze(0))
            
            trainobsactions = [training_action_p1.data.cpu().numpy()[0][0], train_actions[1]]
            trainobs, trainobsreward, trainobsdone, _ = traiobsenv.step(trainobsactions)
            if trainobsdone:
                trainobs = traiobsenv.reset()
            ########################


        if i_update % 100 == 0:                              
            all_rewards1.append(final_rewards1.mean().item())
            all_losses1.append(loss_p1.item())            
            print('step %s' % (i_update))
            print('reward Player1: %s' % np.mean(all_rewards1[-10:]))                                   
            print('loss Player1 %s' % all_losses1[-1])            
            print("---------------------------")
            
            timer.update(time.time())            
            timediff = timer.getTimeDiff()
            total_time = timer.getTotalTime()
            loopstogo = (num_frames - i_update) / 100
            estimatedtimetogo = timer.getTimeToGo(loopstogo)
            logger.printDayFormat("runntime last epochs: ", timediff)
            logger.printDayFormat("total runtime: ", total_time)
            logger.printDayFormat("estimated time to run: ", estimatedtimetogo)           
            print("######## {0} ########".format(sys.argv[1]))
        rollout1.after_update() # player1
       
        if i_update % 1000 == 0 and i_update > 0:
            logger.log(all_rewards1, "Data/", "all_rewards_p1_{0}_{1}.txt".format(sys.argv[1], swich_variable))  
            logger.log(all_losses1, "Data/", "all_losses_p1_{0}_{1}.txt".format(sys.argv[1], swich_variable))      
            logger.log_state_dict(agent1.state_dict(), "Data/agents/agent1_{0}_{1}".format(sys.argv[1], swich_variable))               
            swich_variable += 1
            swich_variable %= 2

    logger.log(all_rewards1, "Data/", "all_rewards_p1_{0}_{1}.txt".format(sys.argv[1], swich_variable))  
    logger.log(all_losses1, "Data/", "all_losses_p1_{0}_{1}.txt".format(sys.argv[1], swich_variable))      
    logger.log_state_dict(agent1.state_dict(), "Data/agents/agent1_{0}_{1}".format(sys.argv[1], swich_variable))        
