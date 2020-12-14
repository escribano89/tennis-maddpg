# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:09:54 2020

@author: Javier Escribano
"""
from unity_env import init_environment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt

from agent import DDPG

# Unity env executable path
UNITY_EXE_PATH = 'Reacher.exe'
# Environment Goal
GOAL = 30.1
# Averaged score
SCORE_AVERAGED = 100
# Let us know the progress each 10 timesteps
PRINT_EVERY = 10
# Number of episode for training
N_EPISODES = 160
# Max Timesteps
MAX_TIMESTEPS = 1000

# Init the reacher environment and get agents, state and action info
env, brain_name, n_agents, state_size, action_size = init_environment(UNITY_EXE_PATH)
agent = DDPG(state_size=state_size, action_size=action_size, random_seed=89)

#  Method for training the agent
def train(n_episodes=N_EPISODES):
    scores_deque = deque(maxlen=SCORE_AVERAGED)
    global_scores = []
    averaged_scores = []
    
    for episode in range(1, N_EPISODES + 1):
        # Get the current states for each agent
        states = env.reset(train_mode=True)[brain_name].vector_observations 
        # Init the score of each agent to zeros
        scores = np.zeros(n_agents)                

        for t in range(MAX_TIMESTEPS):

            # Act according to our policy
            actions = agent.act(states)
            # Send the decided actions to all the agents
            env_info = env.step(actions)[brain_name]        
            # Get next state for each agent
            next_states = env_info.vector_observations     
            # Get rewards obtained from each agent
            rewards = env_info.rewards           
            # Info about if an env is done
            dones = env_info.local_done   
            # Learn from the collected experience
            agent.step(states, actions, rewards, next_states, dones, t)
            # Update current states
            states = next_states   
            # Add the rewards recieved
            scores += rewards    
            
            # Stop the loop if an agent is done               
            if np.any(dones):                          
                break
        
        # Calculate scores and averages
        score = np.mean(scores)
        scores_deque.append(score)
        avg_score = np.mean(scores_deque)
        
        global_scores.append(score)
        averaged_scores.append(avg_score)
                
        if episode % PRINT_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))  
            
        if avg_score >= GOAL:  
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, avg_score))
            torch.save(agent.actor_regular.state_dict(), 'actor_theta.pth')
            torch.save(agent.critic_regular.state_dict(), 'critic_theta.pth')
            break
            
    return global_scores, averaged_scores

# Train the agent and get the results
scores, averages = train()

# Plot Statistics (Global scores and averaged scores)
plt.subplot(2, 1, 2)
plt.plot(np.arange(1, len(scores) + 1), averages)
plt.ylabel('Reacher Environment Average Score')
plt.xlabel('Episode #')
plt.show()