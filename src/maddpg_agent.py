import numpy as np
import random
from src.ddpg_agent import Agent

import torch
import torch.nn.functional as F

import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MAgent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        num_agents,
        random_seed,
        actor_local_load_filenames=[],
        actor_target_load_filenames=[],
        critic_local_load_filenames=[],
        critic_target_load_filenames=[],
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            actor_local_load_filenames   : if given, the initial weights
                                           of the local NN
            critic_local_load_filenames  : if given, the initial weights
                                           of the target NN
            actor_target_load_filenames  : if given, the initial weights
                                           of the local  NN
            critic_target_load_filenames : if given, the initial weights
                                           of the target NN
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
            actor_local_load_filename = (
                actor_local_load_filenames[idx]
                if len(actor_local_load_filenames) == (idx + 1)
                else None
            )
            actor_target_load_filename = (
                actor_target_load_filenames[idx]
                if len(actor_target_load_filenames) == (idx + 1)
                else None
            )
            critic_target_load_filename = (
                critic_target_load_filenames[idx]
                if len(critic_target_load_filenames) == (idx + 1)
                else None
            )
            critic_local_load_filename = (
                critic_local_load_filenames[idx]
                if len(critic_local_load_filenames) == (idx + 1)
                else None
            )

            agent = Agent(
                self.state_size,
                self.action_size,
                self.observed_state_size,
                self.observed_action_size,
                random_seed,
                actor_local_load_filename,
                actor_target_load_filename,
                critic_local_load_filename,
                critic_target_load_filename,
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

            agent.t_step = agent.t_step % cfg.UPDATE_EVERY
            if agent.t_step == 0 and len(agent.memory) > cfg.BATCH_SIZE:
                for _ in range(cfg.NUM_UPDATES):
                    # Sample the memory of the agent
                    experiences = agent.memory.sample(cfg.BATCH_SIZE)
                    # Learning involves knowing the actor of all agents
                    self.learn(experiences, cfg.GAMMA, agent)

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
            x = np.arange(idx, (cfg.BATCH_SIZE * self.num_agents), self.num_agents)
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
            x = np.arange(idx, (cfg.BATCH_SIZE * self.num_agents), self.num_agents)
            actions_pred_list.append(agent.actor_local(observations[x]))

        actions_pred = torch.cat(actions_pred_list, dim=1)
        actor_loss = -train_agent.critic_local(obs, actions_pred).mean()
        # Minimize the loss
        train_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        train_agent.actor_optimizer.step()

        # ----------------------- update target networks ---------------------#
        train_agent.soft_update(
            train_agent.critic_local, train_agent.critic_target, cfg.TAU
        )
        train_agent.soft_update(
            train_agent.actor_local, train_agent.actor_target, cfg.TAU
        )

    def save(
        self,
        actor_local_save_filenames,
        actor_target_save_filenames,
        critic_local_save_filenames,
        critic_target_save_filenames,
    ):
        for idx, agent in enumerate(self.agents):
            agent.save(
                actor_local_save_filenames[idx],
                actor_target_save_filenames[idx],
                critic_local_save_filenames[idx],
                critic_target_save_filenames[idx],
            )
