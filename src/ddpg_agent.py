import numpy as np
import random
import copy
import pdb
from collections import namedtuple, deque

from src.model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 2          # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
UPDATE_EVERY = 1
NUM_UPDATES = 5
EPSILON = 1.0
EPSILON_DECAY = 0.9999
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 observed_state_size,
                 observed_action_size,
                 random_seed,
                 actor_local_load_filename=None,
                 actor_target_load_filename=None,
                 critic_local_load_filename=None,
                 critic_target_load_filename=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            observed_state_size(int): dimension of the states of all agents
            observed_action_size(int): dimension of the actions of all agents
            random_seed (int): random seed
            actor_local_load_filename   : if given, the initial weights of the local  NN
            critic_local_load_filename  : if given, the initial weights if the target NN
            actor_target_load_filename  : if given, the initial weights of the local  NN
            critic_target_load_filename : if given, the initial weights if the target NN
        """
        self.state_size = state_size
        self.action_size = action_size
        self.observed_state_size = observed_state_size
        self.observed_action_size = observed_action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            observed_state_size, observed_action_size, random_seed).to(device)
        self.critic_target = Critic(
            observed_state_size, observed_action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY

    def step(self, states, actions, rewards, next_states, dones, t_step):
        pass

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.epsilon = self.epsilon_decay * self.epsilon
            action += (self.epsilon * self.noise.sample())
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        pass
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self,
             actor_local_save_filename,
             actor_target_save_filename,
             critic_local_save_filename,
             critic_target_save_filename):
        torch.save(self.actor_local.state_dict(),
                   actor_local_save_filename)
        torch.save(self.actor_target.state_dict(),
                   actor_target_save_filename)
        torch.save(self.critic_local.state_dict(),
                   critic_local_save_filename)
        torch.save(self.critic_target.state_dict(),
                   critic_target_save_filename)

    def load(self,
             actor_local_load_filename,
             actor_target_load_filename=None,
             critic_local_load_filename=None,
             critic_target_load_filename=None):
        if actor_local_load_filename is not None:
            self.actor_local.load_state_dict(
                torch.load(actor_local_load_filename))
        if actor_target_load_filename is not None:
            self.actor_target.load_state_dict(
                torch.load(actor_target_load_filename))
        if critic_local_load_filename is not None:
            self.critic_local.load_state_dict(
                torch.load(critic_local_load_filename))
        if critic_target_load_filename is not None:
            self.critic_target.load_state_dict(
                torch.load(critic_target_load_filename))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
            "state", "action", "reward",
            "next_state", "done",
            "observations", "observed_actions", "next_observations"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done,
            observations, observed_actions, next_observations):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward,
                            next_state, done,
                            observations, observed_actions, next_observations)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        observations = torch.from_numpy(np.vstack(
            [e.observations for e in experiences if e is not None])).float().to(device)
        observed_actions = torch.from_numpy(np.vstack(
            [e.observed_actions for e in experiences if e is not None])).float().to(device)
        next_observations = torch.from_numpy(np.vstack(
            [e.next_observations for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones,
                observations, observed_actions, next_observations)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
