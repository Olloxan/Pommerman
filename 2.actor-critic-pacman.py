import sys
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from common.multiprocessing_env import SubprocVecEnv
from common.minipacman import MiniPacman
from common.actor_critic import ActorCritic_Pacman, RolloutStorage
from common.logger import Logger
from common.myTimer import myTimer

import pickle

logger = Logger()
timer = myTimer()

#sys.argv.append("debug")

try:  
    print(sys.argv[1])
except AssertionError:
        print("no Arguments....")

USE_CUDA = torch.cuda.is_available()
MODES = ('regular', 'avoid', 'hunt', 'ambush', 'rush')
mode = 'regular'
num_envs = 8

print(mode)

logger = Logger()

def make_cuda(input):
    if USE_CUDA:
        return input.cuda()
    return input

def make_env():   
    def _thunk():
        env = MiniPacman(mode, 1000)        
        return env
    return _thunk

if __name__ == '__main__': #son windows ding mit mehreren Subprozessen.....
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)    

    state_shape = envs.observation_space.shape
    num_actions = envs.action_space.n

    
    
    #a2c hyperparams:
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    num_steps = 5
    num_frames = int(1e6)

    #rmsprop hyperparams:
    lr    = 7e-4
    eps   = 1e-5
    alpha = 0.99

    #Init a2c and rmsprop
    actor_critic = ActorCritic_Pacman(state_shape, num_actions)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
    
    actor_critic = make_cuda(actor_critic)

    rollout = RolloutStorage(num_steps, num_envs, state_shape)
    if USE_CUDA:        
        rollout.cuda()

    all_rewards = []
    all_losses  = []

    state = envs.reset() 
    
    state = torch.FloatTensor(np.float32(state))

    rollout.states[0].copy_(state)

    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards   = torch.zeros(num_envs, 1)    

    timer.update(time.time())

    for i_update in range(num_frames):

        for step in range(num_steps):                             
            action = actor_critic.act(make_cuda(state))
                               
            next_state, reward, finished, _ = envs.step(action.squeeze(1).cpu().data.numpy())
               
            reward = torch.FloatTensor(reward).unsqueeze(1)
            episode_rewards += reward
            finished_masks = torch.FloatTensor(1-np.array(finished)).unsqueeze(1)                                                       

            final_rewards *= finished_masks
            final_rewards += (1-finished_masks) * episode_rewards                       
                                                              
            episode_rewards *= finished_masks
                                                                       
            finished_masks = make_cuda(finished_masks)

            state = torch.FloatTensor(np.float32(next_state))
            rollout.insert(step, state, action.data, reward, finished_masks)


        _, next_value = actor_critic(rollout.states[-1])
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, gamma)

        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
            rollout.states[:-1].view(-1, *state_shape),
            rollout.actions.view(-1, 1)
        )

        values = values.view(num_steps, num_envs, 1)
        action_log_probs = action_log_probs.view(num_steps, num_envs, 1)        
        advantages = returns - values

        value_loss = advantages.pow(2).mean()        
        action_loss = -(advantages.data * action_log_probs).mean()

        optimizer.zero_grad()        
        loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
        
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
        optimizer.step()
    
        if i_update % 100 == 0:            
            all_rewards.append(final_rewards.mean().item())
            all_losses.append(loss.item())
            print('epoch %s. reward: %s' % (i_update, np.mean(all_rewards[-10:])))            
            print('loss %s' % all_losses[-1])
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
        rollout.after_update()
        
        if i_update % 5000 == 0 and i_update > 0:
            logger.log(all_rewards, "Data/", "all_rewards_{0}_{1}.txt".format(sys.argv[1], mode))  
            logger.log(all_losses, "Data/", "all_losses_{0}_{1}.txt".format(sys.argv[1], mode))      
            logger.log_state_dict(actor_critic.state_dict(), "Data/agents/actor_critic_{0}_{1}".format(sys.argv[1], mode))    

    logger.log(all_rewards, "Data/", "all_rewards_{0}_{1}.txt".format(sys.argv[1], mode))  
    logger.log(all_losses, "Data/", "all_losses_{0}_{1}.txt".format(sys.argv[1], mode))      
    logger.log_state_dict(actor_critic.state_dict(), "Data/agents/actor_critic_{0}_{1}".format(sys.argv[1], mode))    

    

