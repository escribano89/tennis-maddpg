import random
from actor import Actor
from critic import Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import ReplayBuffer

# Replay Buffer Size
BUFFER_SIZE = int(1e6)
# Minibatch Size
BATCH_SIZE = 256 
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG():
    """Interacts with and learns from the environment using the DDPG algorithm."""

    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Neural Network (Regular and target)
        self.actor_regular = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_regular.parameters(), lr=LR_ACTOR)

        # Critic Neural Network (Regular and target)
        self.critic_regular = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_regular.parameters(), lr=LR_CRITIC)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
          
        # Ensure that both networks have the same weights
        self.deep_copy(self.actor_target, self.actor_regular)
        self.deep_copy(self.critic_target, self.critic_regular)

    def step(self, states, actions, rewards, next_states, dones, timestep):
        # Save collected experiences
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn from our buffer if possible
        if len(self.memory) > BATCH_SIZE and timestep % UPDATE_EVERY == 0:
            for _ in range(N_EXPERIENCES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states):
        states = torch.from_numpy(states).float().to(device)
        
        # Evaluation mode
        # Notify all your layers that you are in eval mode, that way, 
        # Batchnorm or dropout layers will work in eval mode instead of training mode.
        self.actor_regular.eval()
        # torch.no_grad() impacts the autograd engine and deactivate it. 
        # It will reduce memory usage and speed up
        with torch.no_grad():
            actions = self.actor_regular(states).cpu().data.numpy()
        # Enable Training mode
        self.actor_regular.train()

        return actions


    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Update the critic neural network
        # Get predicted next-state actions
        actions_next = self.actor_target(next_states)
        #▓Get Q values from target model
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Calculate the critic loss
        Q_expected = self.critic_regular(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor neural network
        # Calculate the actor loss
        actions_pred = self.actor_regular(states)
        # Change sign because of the gradient descent
        actor_loss = -self.critic_regular(states, actions_pred).mean()

        # Minimize the loss function
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target network using the soft update approach (slowly updating)
        self.soft_update(self.critic_regular, self.critic_target, TAU)
        self.soft_update(self.actor_regular, self.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        # Update the target network slowly to improve the stability
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau) * target_param.data)

    def deep_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


