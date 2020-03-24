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

from common.actor_critic import ActorCritic, RolloutStorage
from common.multiprocessing_env import SubprocVecEnv
from common.logger import Logger
from common.myTimer import myTimer

sys.argv.append("AC_Pommer_newLoss_64x1e6x5")

logger = Logger()
timer = myTimer()

USE_CUDA = torch.cuda.is_available()

# Instantiate the environment
config = ffa_v0_fast_env()

num_envs = 64
num_players = 2

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
    def _thunk():
        env = Pomme(**config["env_kwargs"])
        agents = {}
        for agent_id in range(num_players):
            agent = TrainingAgent(config["agent"](agent_id, config["game_type"]))              
            agents[agent_id] = agent
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
    agent1 = ActorCritic(state_shape, num_actions)
    agent2 = ActorCritic(state_shape, num_actions)
    optimizer1 = optim.RMSprop(agent1.parameters(), lr, eps=eps, alpha=alpha)
    optimizer2 = optim.RMSprop(agent2.parameters(), lr, eps=eps, alpha=alpha)

    agent1 = make_cuda(agent1)
    agent2 = make_cuda(agent2)

    rollout1 = RolloutStorage(num_steps, num_envs, state_shape)
    rollout2 = RolloutStorage(num_steps, num_envs, state_shape)

    if USE_CUDA:        
        rollout1.cuda()
        rollout2.cuda()

    all_rewards1 = []
    all_losses1  = []

    all_rewards2 = []
    all_losses2  = []

    observations = envs.reset() 

    state = featurize(observations)
    state = make_cuda(torch.FloatTensor(state))

    rollout1.states[0].copy_(state)
    rollout2.states[0].copy_(state)

    episode_rewards1 = torch.zeros(num_envs, 1)
    final_rewards1   = torch.zeros(num_envs, 1)    

    episode_rewards2 = torch.zeros(num_envs, 1)
    final_rewards2   = torch.zeros(num_envs, 1)    

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
            
            # actor2 acts in all parallel envs
            action_p2 = agent2.act(make_cuda(state)).squeeze(1).cpu().numpy() 

            # separate actions
            action_tuples = []              
            for i in range(num_envs):
                actions = []
                actions.append(action_p1[i])    # player1
                actions.append(action_p2[i])    # player2
                action_tuples.append(actions)
            
            next_observation, reward, finished, _ = envs.step(action_tuples)    # pass actions to environments

            # separate rewards
            reward1 = []                    
            reward2 = []
            for i in range(num_envs):
                reward1.append(reward[i][player0]) # player1
                reward2.append(reward[i][player1]) # player2

            reward1 = torch.FloatTensor(reward1).unsqueeze(1) # player1
            reward2 = torch.FloatTensor(reward2).unsqueeze(1) # player2
            episode_rewards1 += reward1 # player1
            episode_rewards2 += reward2 # player2

            finished_masks = torch.FloatTensor(1-np.array(finished)).unsqueeze(1)                                                       

            # final rewards player1
            final_rewards1 *= finished_masks
            final_rewards1 += (1-finished_masks) * episode_rewards1                       
            
            # final rewards player2
            final_rewards2 *= finished_masks
            final_rewards2 += (1-finished_masks) * episode_rewards2

            episode_rewards1 *= finished_masks # player1
            episode_rewards2 *= finished_masks # player2
                                                                       
            finished_masks = make_cuda(finished_masks)

            state = make_cuda(torch.FloatTensor(np.float32(featurize(next_observation))))
            rollout1.insert(step, state, torch.FloatTensor(action_p1).unsqueeze(1), reward1, finished_masks) # player1
            rollout2.insert(step, state, torch.FloatTensor(action_p2).unsqueeze(1), reward2, finished_masks) # player2

        # v(s_t+1) player1
        _, next_value_p1 = agent1(rollout1.states[-1])
        next_value_p1 = next_value_p1.data

        # v(s_t+1) player2
        _, next_value_p2 = agent2(rollout2.states[-1])
        next_value_p2 = next_value_p2.data

        # n-step returns player1
        returns_p1 = rollout1.compute_returns(next_value_p1, gamma)

        # n-step returns player2
        returns_p2 = rollout2.compute_returns(next_value_p2, gamma)

        # eval actions player1
        logit_p1, action_log_probs_p1, values_p1, entropy_p1 = agent1.evaluate_actions(
            rollout1.states[:-1].view(-1, *state_shape),
            rollout1.actions.view(-1, 1)
        )

        # eval actions player2
        logit_p2, action_log_probs_p2, values_p2, entropy_p2 = agent2.evaluate_actions(
            rollout2.states[:-1].view(-1, *state_shape),
            rollout2.actions.view(-1, 1)
        )

        # advantages player1
        values_p1 = values_p1.view(num_steps, num_envs, 1)
        action_log_probs_p1 = action_log_probs_p1.view(num_steps, num_envs, 1)        
        advantages_p1 = returns_p1 - values_p1

        value_loss_p1 = advantages_p1.pow(2).mean()        
        action_loss_p1 = -(advantages_p1.data * action_log_probs_p1).mean()

        # advantages player2
        values_p2 = values_p2.view(num_steps, num_envs, 1)
        action_log_probs_p2 = action_log_probs_p2.view(num_steps, num_envs, 1)        
        advantages_p2 = returns_p2 - values_p2

        value_loss_p2 = advantages_p2.pow(2).mean()        
        action_loss_p2 = -(advantages_p2.data * action_log_probs_p2).mean()

        # optimize player1
        optimizer1.zero_grad()        
        loss_p1 = value_loss_p1 * value_loss_coef + action_loss_p1 - entropy_p1 * entropy_coef
        
        loss_p1.backward()
        nn.utils.clip_grad_norm_(agent1.parameters(), max_grad_norm)
        optimizer1.step()
        
        # optimize player2
        optimizer2.zero_grad()        
        loss_p2 = value_loss_p2 * value_loss_coef + action_loss_p2 - entropy_p2 * entropy_coef
        
        loss_p2.backward()
        nn.utils.clip_grad_norm_(agent2.parameters(), max_grad_norm)
        optimizer2.step()

        if i_update % 5 == 0:            
             # training observation #
            traiobsenv.render()        
            trainobs_state = make_cuda(torch.FloatTensor(np.float32(featureize_single(trainobs[0]))))
            training_action_p1 = agent1.act(trainobs_state.unsqueeze(0))
            training_action_p2 = agent2.act(trainobs_state.unsqueeze(0))
            trainobsactions = [training_action_p1.data.cpu().numpy()[0][0], training_action_p2.data.cpu().numpy()[0][0]]
            trainobs, trainobsreward, trainobsdone, _ = traiobsenv.step(trainobsactions)
            if trainobsdone:
                trainobs = traiobsenv.reset()
            ########################


        if i_update % 100 == 0:                              
            all_rewards1.append(final_rewards1.mean().item())
            all_losses1.append(loss_p1.item())
            all_rewards2.append(final_rewards2.mean().item())
            all_losses2.append(loss_p2.item())
            print('step %s' % (i_update))
            print('reward Player1: %s' % np.mean(all_rewards1[-10:]))                        
            print('reward Player2: %s' % np.mean(all_rewards2[-10:]))            
            print('loss Player1 %s' % all_losses1[-1])
            print('loss Player2 %s' % all_losses2[-1])
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
        rollout2.after_update() # player2

        if i_update % 1000 == 0 and i_update > 0:
            logger.log(all_rewards1, "Data/", "all_rewards_p1_{0}_{1}.txt".format(sys.argv[1], swich_variable))  
            logger.log(all_losses1, "Data/", "all_losses_p1_{0}_{1}.txt".format(sys.argv[1], swich_variable))      
            logger.log_state_dict(agent1.state_dict(), "Data/agents/agent1_{0}_{1}".format(sys.argv[1], swich_variable))    
            logger.log(all_rewards2, "Data/", "all_rewards_p2_{0}_{1}.txt".format(sys.argv[1], swich_variable))  
            logger.log(all_losses2, "Data/", "all_losses_p2_{0}_{1}.txt".format(sys.argv[1], swich_variable))      
            logger.log_state_dict(agent2.state_dict(), "Data/agents/agent2_{0}_{1}".format(sys.argv[1], swich_variable))
            swich_variable += 1
            swich_variable %= 2

    logger.log(all_rewards1, "Data/", "all_rewards_p1_{0}_{1}.txt".format(sys.argv[1], swich_variable))  
    logger.log(all_losses1, "Data/", "all_losses_p1_{0}_{1}.txt".format(sys.argv[1], swich_variable))      
    logger.log_state_dict(agent1.state_dict(), "Data/agents/agent1_{0}_{1}".format(sys.argv[1], swich_variable))    
    logger.log(all_rewards2, "Data/", "all_rewards_p2_{0}_{1}.txt".format(sys.argv[1], swich_variable))  
    logger.log(all_losses2, "Data/", "all_losses_p2_{0}_{1}.txt".format(sys.argv[1], swich_variable))      
    logger.log_state_dict(agent2.state_dict(), "Data/agents/agent2_{0}_{1}".format(sys.argv[1], swich_variable))