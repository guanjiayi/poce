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
"""Offline Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch

from omnisafe.common.logger import Logger
from omnisafe.envs.core import make, support_envs
from omnisafe.envs.wrapper import ActionScale, TimeLimit
from omnisafe.models.base import Actor
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device


class OfflineAdapter_AUG:
    """Offline Adapter for OmniSafe.

    :class:`OfflineAdapter` is used to adapt the environment to the offline training.

    .. note::
        Technically, Offline training doesn't need env to interact with the agent.
        However, to visualize the performance of the agent when training,
        we still need instantiate a environment to evaluate the agent.
        OfflineAdapter provide an important interface ``evaluate`` to test the agent.

    Args:
        env_id (str): The environment id.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OfflineAdapter`."""
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._env_id = env_id
        self._env = make(env_id, num_envs=1, device=cfgs.train_cfgs.device)
        self._cfgs = cfgs
        self._device = get_device(cfgs.train_cfgs.device)

        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, 1000, device=self._device)
        self._env = ActionScale(self._env, device=self._device, high=1.0, low=-1.0)

        self._env.set_seed(seed)

    @property
    def action_space(self) -> OmnisafeSpace:
        """The action space of the environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> OmnisafeSpace:
        """The observation space of the environment."""
        return self._env.observation_space

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        return self._env.step(actions)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        return self._env.reset(seed=seed, options=options)

    def evaluate(
        self,
        evaluate_epoisodes: int,
        agent: Actor,
        logger: Logger,
    ) -> None:
        """Evaluate the agent in the environment.

        Args:
            evaluate_epoisodes (int): the number of episodes for evaluation.
            agent (Actor): the agent to be evaluated.
            logger (Logger): the logger for logging the evaluation results.
        """
        for _ in range(evaluate_epoisodes):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0.0

            done = torch.Tensor([False])
            obs, _ = self.reset()
            while not done:
                action = agent.predict(obs.unsqueeze(0), deterministic=True)
                obs, reward, cost, terminated, truncated, _ = self.step(action.squeeze(0))

                ep_ret += reward.item()
                ep_cost += cost.item()
                ep_len += 1

                done = torch.logical_or(terminated, truncated)

            logger.store(
                {
                    'Metrics/EpRet': ep_ret,
                    'Metrics/EpCost': ep_cost,
                    'Metrics/EpLen': ep_len,
                },
            )


    def evaluate_baseline_bcq(
        self,
        evaluate_epoisodes: int,
        agent: Actor,
        logger: Logger,
    ) -> None:
        """Evaluate the agent in the environment.

        Args:
            evaluate_epoisodes (int): the number of episodes for evaluation.
            agent (Actor): the agent to be evaluated.
            logger (Logger): the logger for logging the evaluation results.
        """
        for _ in range(evaluate_epoisodes):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0.0
            obs, _ = self.reset()
            M_cur = torch.tensor([0.]).to(self._device)
            obs_aug = torch.cat((obs, M_cur))
            obs_aug = obs_aug.to(self._device)
            first_step: bool = True

            # done = False
            done = torch.Tensor([False])

            cost_max:float = 0.0
            cost_rate:float = 0.0

            while not done:
                act = agent.predict(obs_aug.unsqueeze(0), deterministic=True)
                # act = agent.actor.predict(obs_aug, deterministic=True)
                next_obs, reward, cost, terminated, truncated, info = self.step(act.squeeze())
                next_obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (next_obs, reward, cost, terminated, truncated)
                )

                if first_step:
                    cost_increase = info.get('original_cost', cost).reshape(1)
                    M_next = info.get('original_cost', cost).reshape(1)   
                    first_step = False
                else:
                    cost_increase = max(info.get('original_cost', cost)-M_cur, torch.tensor([0.]).to(self._device))
                    M_next = M_cur + cost_increase

                # record the cost rate
                if info.get('original_cost', cost).cpu().reshape(1).numpy() >= self._cfgs.algo_cfgs.cost_increase_limit:
                    cost_rate +=1

                # print('cost:', info.get('original_cost', cost).cpu().reshape(1).numpy() )
                # print('cost_rate:', cost_rate)

                # recore the max cost
                if info.get('original_cost', cost).cpu().reshape(1).numpy() > cost_max:
                    cost_max = info.get('original_cost', cost).cpu().reshape(1).numpy()

                next_obs_aug = torch.cat((next_obs, M_next))

                ep_ret += info.get('original_reward', reward).cpu().numpy()
                ep_cost += info.get('original_cost', cost).cpu().numpy()
                ep_len += 1

                obs_aug = next_obs_aug
                M_cur = M_next
                # done = bool(terminated[0].item()) or bool(truncated[0].item())
    
                done = torch.logical_or(terminated, truncated)
                if ep_len >= self._cfgs.train_cfgs.ep_step:
                    done = torch.tensor([True]).to(self._device)

            logger.store(
                {
                    'Metrics/EpRet': ep_ret,
                    'Metrics/EpCost': ep_cost,
                    'Metrics/EpLen': ep_len,
                    'Metrics/TestCostRate': cost_rate/ep_len,
                    'Metrics/TestCostMax': cost_max,
                    'Metrics/TestM_cur': M_cur.cpu().numpy(),
                },
            )
    

    def evaluate_offaug(
        self,
        evaluate_epoisodes: int,
        agent: Actor,
        logger: Logger,
    ) -> None:
        """Evaluate the agent in the environment.

        Args:
            evaluate_epoisodes (int): the number of episodes for evaluation.
            agent (Actor): the agent to be evaluated.
            logger (Logger): the logger for logging the evaluation results.
        """
        for _ in range(evaluate_epoisodes):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0.0
            obs, _ = self.reset()
            M_cur = torch.tensor([0.]).to(self._device)
            obs_aug = torch.cat((obs, M_cur))
            obs_aug = obs_aug.to(self._device)
            first_step: bool = True

            # done = False
            done = torch.Tensor([False])

            cost_max:float = 0.0
            cost_rate:float = 0.0

            while not done:
                act = agent.actor.predict(obs_aug, deterministic=True)
                next_obs, reward, cost, terminated, truncated, info = self.step(act)
                next_obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (next_obs, reward, cost, terminated, truncated)
                )

                # if first_step:
                #     cost_increase = info.get('original_cost', cost).cpu().reshape(1)
                #     M_next = info.get('original_cost', cost).cpu().reshape(1)   
                #     first_step = False
                # else:
                #     cost_increase = max(info.get('original_cost', cost).cpu()-M_cur, torch.tensor([0.]))
                #     M_next = M_cur + cost_increase

                if first_step:
                    cost_increase = info.get('original_cost', cost).reshape(1)
                    M_next = info.get('original_cost', cost).reshape(1)   
                    first_step = False
                else:
                    cost_increase = max(info.get('original_cost', cost)-M_cur, torch.tensor([0.]).to(self._device))
                    M_next = M_cur + cost_increase

                # record the cost rate
                if info.get('original_cost', cost).cpu().reshape(1).numpy() >= self._cfgs.algo_cfgs.cost_increase_limit:
                    cost_rate +=1

                # recore the max cost
                if info.get('original_cost', cost).cpu().reshape(1).numpy() > cost_max:
                    cost_max = info.get('original_cost', cost).cpu().reshape(1).numpy().item()

                # print('cost_max_current:', info.get('original_cost', cost).cpu().reshape(1).numpy())
                # print('cost_max:', cost_max)

                next_obs_aug = torch.cat((next_obs, M_next))

                ep_ret += info.get('original_reward', reward).cpu().numpy()
                ep_cost += info.get('original_cost', cost).cpu().numpy()
                ep_len += 1

                obs_aug = next_obs_aug
                M_cur = M_next
                # done = bool(terminated[0].item()) or bool(truncated[0].item())
    
                done = torch.logical_or(terminated, truncated)
                if ep_len >= self._cfgs.train_cfgs.ep_step:
                    done = torch.tensor([True]).to(self._device)

            logger.store(
                {
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpCost': ep_cost,
                    'Metrics/TestEpLen': ep_len,
                    'Metrics/TestCostRate': cost_rate/ep_len,
                    'Metrics/TestCostMax': cost_max,
                    'Metrics/TestM_cur': M_cur.cpu().numpy().item(),
                },
            )
