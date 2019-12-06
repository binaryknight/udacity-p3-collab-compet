import numpy as np
import random
import copy
import pdb
from collections import namedtuple, deque

from src.model import Actor, Critic
from src.ddpg_agent import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-5  # learning rate of the actor
LR_CRITIC = 1e-5  # learning rate of the critic
WEIGHT_DECAY = 0.0  # L2 weight decay
UPDATE_EVERY = 5
NUM_UPDATES = 5
EPSILON = 1.0
EPSILON_DECAY = 0.9999
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MAgent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        num_agents,
        random_seed,
        actor_local_load_filename=[],
        actor_target_load_filename=[],
        critic_local_load_filename=[],
        critic_target_load_filename=[],
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            actor_local_load_filename   : if given, the initial weights of the local  NN
            critic_local_load_filename  : if given, the initial weights if the target NN
            actor_target_load_filename  : if given, the initial weights of the local  NN
            critic_target_load_filename : if given, the initial weights if the target NN
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.observed_state_size = num_agents * state_size
        self.observed_action_size = num_agents * action_size
        self.seed = random.seed(random_seed)
        self.agents = []

        # We need generate the DDPG agents
        for idx in range(self.num_agents):
            agent = Agent(
                self.state_size,
                self.action_size,
                self.observed_state_size,
                self.observed_action_size,
                random_seed,
            )
            self.agents.append(agent)

    def step(self, states, actions, rewards, next_states, dones, t_step):
        # Save the experiences and the rewards
        # We need to reshape the states to obtain the observations
        for idx, agent in enumerate(self.agents):
            agent.memory.add(
                states[[idx]],
                actions[[idx]],
                rewards[idx],
                next_states[[idx]],
                dones[idx],
                states,
                actions,
                next_states,
            )

            agent.t_step = agent.t_step % UPDATE_EVERY
            if agent.t_step == 0 and len(agent.memory) > BATCH_SIZE:
                for _ in range(NUM_UPDATES):
                    # Sample the memory of the agent
                    experiences = agent.memory.sample()
                    # Learning involves knowing the actor of all agents
                    self.learn(experiences, GAMMA, agent)

    def act(self, state, add_noise=True):
        actions_list = []
        for idx, agent in enumerate(self.agents):
            actions_list.append(agent.act(state[[idx]], add_noise))
        actions = np.vstack(actions_list)
        return actions

    def learn(self, experiences, gamma, train_agent):
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            observations,
            observed_actions,
            next_observations,
        ) = experiences
        # ---------------------------- update critic -------------------------#
        # Get predicted next-state actions and Q values from target models
        observations_next_list = []
        actions_next_list = []
        observed_actions_list = []
        observations_list = []

        for idx, agent in enumerate(self.agents):
            x = np.arange(idx, (BATCH_SIZE * self.num_agents), self.num_agents)
            actions_next_list.append(agent.actor_target(next_observations[x]))
            observed_actions_list.append(observed_actions[x])
            observations_next_list.append(next_observations[x])
            observations_list.append(observations[x])

        obs_next = torch.cat(observations_next_list, dim=1)
        obs = torch.cat(observations_list, dim=1)
        obs_actions = torch.cat(observed_actions_list, dim=1)
        actions_next = torch.cat(actions_next_list, dim=1)
        Q_targets_next = train_agent.critic_target(obs_next, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = train_agent.critic_local(obs, obs_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        train_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(train_agent.critic_local.parameters(), 1)
        train_agent.critic_optimizer.step()

        # ---------------------------- update actor --------------------------#
        # Compute actor loss
        actions_pred_list = []
        for idx, agent in enumerate(self.agents):
            x = np.arange(idx, (BATCH_SIZE * self.num_agents), self.num_agents)
            actions_pred_list.append(agent.actor_local(observations[x]))

        actions_pred = torch.cat(actions_pred_list, dim=1)
        actor_loss = -train_agent.critic_local(obs, actions_pred).mean()
        # Minimize the loss
        train_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        train_agent.actor_optimizer.step()

        # ----------------------- update target networks ---------------------#
        train_agent.soft_update(
            train_agent.critic_local, train_agent.critic_target, TAU
        )
        train_agent.soft_update(train_agent.actor_local, train_agent.actor_target, TAU)

    def save(self):
        pass

    def load(self):
        pass
