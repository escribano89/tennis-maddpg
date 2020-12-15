# -*- coding: utf-8 -*-

# Replay Buffer Size
BUFFER_SIZE = int(1e6)
# Minibatch Size
BATCH_SIZE = 128 
# Discount Gamma
GAMMA = 0.995 
# Soft Update Value
TAU = 1e-2  
# Learning rates for each NN      
LR_ACTOR = 1e-3 
LR_CRITIC = 1e-3
# Update network every X timesteps
UPDATE_EVERY = 32
# Learn from batch of experiences n_experiences times
N_EXPERIENCES = 16   
# Noise parameters
OU_MU = 0.0
# Volatility
OU_SIGMA = 0.2       
# Speed of mean reversion   
OU_THETA = 0.1 
# Noise start
NOISE_START = 0.5 
# Noise decay
NOISE_DECAY = 0.9 
# Stop noise in this episode
NOISE_EPISODE_STOP = 256
# Noise parameters
OU_MU = 0.0
# Volatility
OU_SIGMA = 0.2       
# Speed of mean reversion   
OU_THETA = 0.1 