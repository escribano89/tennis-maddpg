# -*- coding: utf-8 -*-
import torch
from replay_buffer import ReplayBuffer
from ddpg import DDPG
import numpy as np
from params import  (
    BUFFER_SIZE, BATCH_SIZE, NOISE_EPISODE_STOP, UPDATE_EVERY, NOISE_START, NOISE_DECAY, GAMMA    
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    def __init__(self, state_size, action_size, n_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        self.n_agents = n_agents
        
         # Exploration noise
        self.noise_enabled = False
        self.noise_value = NOISE_START
        self.noise_decay = NOISE_DECAY
        
        # Initiate n DDPG agents
        self.agents = self.setup_agents(n_agents)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def setup_agents(self, n_agents):
        agents = []
        for i in range(n_agents):
            agents.append(DDPG(i, self.state_size, self.action_size, self.n_agents, self.seed))
        return agents

    def step(self, states, actions, rewards, next_states, dones, timestep):
        # Flat states and next states
        states = states.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        # Add experience to the buffer
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # Check if we should stop noise depending on the current timestep
        if timestep >= NOISE_EPISODE_STOP:
            self.noise_enabled = False
        
        # Learn from our buffer if possible
        if len(self.memory) > BATCH_SIZE and timestep % UPDATE_EVERY == 0:
            # Sample experiences for each agent
            experiences = [self.memory.sample() for _ in range(self.n_agents)]
            self.learn(experiences, GAMMA)
                
    
    def act(self, states, add_noise=True):
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, noise_value=self.noise_value, add_noise=self.noise_enabled)
            # Decay noise
            self.noise_value *= self.noise_decay
            actions.append(action)
        # Return flattened actions
        return np.array(actions).reshape(1, -1)
    
    def checkpoint(self):
        # Save actor and critic weights for each agent
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_regular.state_dict(),  "actor_agent_{}.pth".format(i))
            torch.save(agent.critic_regular.state_dict(), "critic_agent_{}.pth".format(i))

    def learn(self, experiences, gamma):
        next_actions = []
        actions = []
        for i, agent in enumerate(self.agents):
            states, _ , _ , next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)            
            state = states.reshape(-1, self.action_size, self.state_size).index_select(1, agent_id).squeeze(1)
            action = agent.actor_regular(state)
            actions.append(action)

            next_state = next_states.reshape(-1, self.action_size, self.state_size).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            next_actions.append(next_action)
                       
        # Call to the method learn for each agent using
        # the related experiences and all actions/next actions
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, next_actions, actions)

    def soft_update(self, local_model, target_model, tau):
        # Update the target network slowly to improve the stability
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau) * target_param.data)

    def deep_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


