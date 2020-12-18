# -*- coding: utf-8 -*-
from unity_env import init_environment
import numpy as np
from maddpg import MADDPG
from params import UNITY_EXE_PATH


# Init the reacher environment and get agents, state and action info
env, brain_name, n_agents, state_size, action_size = init_environment(UNITY_EXE_PATH, train=False)
agent = MADDPG(state_size=state_size, action_size=action_size, n_agents=n_agents, random_seed=89)
# Load trained weights
agent.load_weights()

env_info = env.reset(train_mode=False)[brain_name]     
states = env_info.vector_observations                  
scores = np.zeros(n_agents)   
            
while True:
    actions = agent.act(states)                     
    env_info = env.step(actions)[brain_name]          
    next_states = env_info.vector_observations       
    rewards = env_info.rewards                        
    dones = env_info.local_done                        
    scores += env_info.rewards                         
    states = next_states                              


env.close()