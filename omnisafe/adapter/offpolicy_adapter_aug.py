# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""OffPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch

import numpy as np

from omnisafe.adapter.online_adapter_aug import OnlineAdapter_AUG
from omnisafe.common.buffer import VectorOffPolicyBuffer_AUG
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic_aug import ConstraintActorQCritic_AUG
from omnisafe.utils.config import Config


class OffPolicyAdapter_AUG(OnlineAdapter_AUG):
    """OffPolicy Adapter for OmniSafe.

    :class:`OffPolicyAdapter` is used to adapt the environment to the off-policy training.

    .. note::
        Off-policy training need to update the policy before finish the episode,
        so the :class:`OffPolicyAdapter` will store the current observation in ``_current_obs``.
        After update the policy, the agent will *remember* the current observation and
        use it to interact with the environment.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _current_obs: torch.Tensor
    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OffPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._current_obs, _ = self.reset()
        self._max_ep_len: int = 1000
        # The max cost of the episode
        self._M: torch.Tensor = torch.tensor([0.])
        # The augment obs for the state-wise RL
        # self._aug_current_obs = torch.from_numpy(np.append(self._current_obs.numpy(), self._M))
        self._aug_current_obs = torch.cat((self._current_obs, self._M.unsqueeze(0)), dim=1)
        self._first_step: bool = True
        self._step: int  = 0
        self._reset_log()

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic_AUG,
        logger: Logger,
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorCritic): Agent.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        for _ in range(episode):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            M_cur = torch.tensor([0.])
            aug_obs = torch.cat((obs,M_cur.unsqueeze(0)),dim=1)
            aug_obs = aug_obs.to(self._device)
            first_step: bool = True

            done = False
            while not done:
                self._step += 1
                act = agent.step(aug_obs, deterministic=True)
                next_obs, reward, cost, terminated, truncated, info = self._eval_env.step(act)
                next_obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (next_obs, reward, cost, terminated, truncated)
                )

                if first_step:
                    cost_increase = info.get('original_cost', cost).cpu()
                    M_next = info.get('original_cost', cost).cpu()
                    first_step = False
                else:
                    cost_increase = max(info.get('original_cost', cost).cpu()-M_cur, torch.tensor([0.]))
                    M_next = M_cur + cost_increase

                aug_next_obs = torch.cat((next_obs, M_next.unsqueeze(0)), dim = 1)

                ep_ret += info.get('original_reward', reward).cpu()
                ep_cost += info.get('original_cost', cost).cpu()
                ep_len += 1

                aug_obs = aug_next_obs
                M_cur = M_next
                done = bool(terminated[0].item()) or bool(truncated[0].item())

            logger.store(
                {
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpCost': ep_cost,
                    'Metrics/TestEpLen': ep_len,
                },
            )

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic_AUG,
        buffer: VectorOffPolicyBuffer_AUG,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """
        for _ in range(rollout_step):
            self._step += 1
            if use_rand_action:
                act = torch.as_tensor(self._env.sample_action(), dtype=torch.float32).to(
                    self._device,
                )
            else:
                act = agent.step(self._aug_current_obs, deterministic=False)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            # For the state-wise RL
            if self._first_step:
                cost_increase = info.get('original_cost', cost).cpu()
                M_next = info.get('original_cost', cost).cpu()
                self._first_step = False
            else:
                cost_increase = max(info.get('original_cost', cost).cpu()-self._M, torch.tensor([0.]))
                M_next = self._M + cost_increase

            self._log_value(reward=reward, cost=cost, info=info)
            real_next_obs = next_obs.clone()
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    if 'final_observation' in info:
                        real_next_obs[idx] = info['final_observation'][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)
            

            # self._aug_real_next_obs = np.append(real_next_obs, M_next)
            self._aug_real_next_obs = torch.cat((real_next_obs, M_next.unsqueeze(0)), dim=1)
            buffer.store(
                obs = self._aug_current_obs,
                act = act,
                reward = reward,
                cost = cost,
                done = torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                next_obs = self._aug_real_next_obs,
                # cost_inc=torch.tensor(cost_increase, dtype=torch.float32).to(self._device),
                cost_increase = cost_increase,
            )

            # print('step:', self._step)
            # print('cost_increase:', cost_increase.item())
            # print('self._M:', self._M.item())
            # print('cost:', cost.item())

            # self._current_obs = next_obs
            self._aug_current_obs = self._aug_real_next_obs
            self._M = M_next

            # Reset the parameter for the state-wise
            if torch.logical_or(terminated, truncated):
                self._M = torch.tensor([0.])
                self._aug_current_obs = torch.cat((self._current_obs, self._M.unsqueeze(0)), dim=1)
                self._first_step = True

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
