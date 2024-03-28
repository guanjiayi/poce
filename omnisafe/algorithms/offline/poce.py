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

from copy import deepcopy
from typing import Tuple

import torch
from torch import nn, optim

from torch.nn.utils.clip_grad import clip_grad_norm_
from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.poce_core import POCE_CORE
from omnisafe.common.lagrange import Lagrange
from omnisafe.models.critic.critic_builder_aug import CriticBuilder_AUG
from omnisafe.common.vae_caug import VAE_CAUG
from torch.distributions import Normal


@registry.register
class POCE(POCE_CORE):
    """Constraint variant of CRR.

    References:
        - Title: COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation
        - Author: Lee, JongminPaduraru, CosminMankowitz, Daniel JHeess, NicolasPrecup, Doina
        - URL: `https://arxiv.org/abs/2204.08957`
    """

    def _init(self) -> None:
        """Initialize an instance of :class:`CCRR_AUG`."""

        super()._init()
        self._rew_update: int = 0
        self._cost_update: int = 0
        self._cost_increase_update: int = 0
        self._step : int = 0

        # CQL parameters
        self._num_random = 10
        self._temp = 1.0
        self._min_q_weight = 1.0
        self._target_action_gap = 10.0

        if self._cfgs.algo_cfgs.auto_alpha:
            self._target_entropy = -torch.prod(torch.Tensor(self._env.action_space.shape)).item()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)

            assert self._cfgs.model_cfgs.critic.lr is not None
            self._alpha_optimizer = optim.Adam(
                [self._log_alpha],
                lr=self._cfgs.model_cfgs.critic.lr,
            )
        else:
            self._log_alpha = torch.log(
                torch.tensor(self._cfgs.algo_cfgs.alpha, device=self._device),
            )

    @property
    def _alpha(self) -> float:
        """The value of alpha."""
        return self._log_alpha.exp().item()

    def _init_log(self) -> None:
        """Log the C-CRR specific information.

        +----------------------------+---------------------------------------------------------+
        | Things to log              | Description                                             |
        +============================+=========================================================+
        | Loss/Loss_cost_critic      | Loss of the cost critic.                                |
        +----------------------------+---------------------------------------------------------+
        | Qc/data_Qc                 | Average cost Q value of offline data.                   |
        +----------------------------+---------------------------------------------------------+
        | Qc/target_Qc               | Average cost Q value of next_obs and next_action.       |
        +----------------------------+---------------------------------------------------------+
        | Qc/current_Qc              | Average cost Q value of obs and agent predicted action. |
        +----------------------------+---------------------------------------------------------+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier.                                |
        +----------------------------+---------------------------------------------------------+
        """
        super()._init_log()

        self._logger.register_key('Loss/Loss_cost_critic')
        self._logger.register_key('Qc/data_Qc')
        self._logger.register_key('Qc/target_Qc')
        self._logger.register_key('Qc/current_Qc')
        self._logger.register_key('Loss/Loss_cost_critic_increase')
        self._logger.register_key('Qc/data_Qc_increase')
        self._logger.register_key('Qc/target_Qc_increase')
        self._logger.register_key('Qc/current_Qc_increase')

        # self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Metrics/TestCostRate')
        self._logger.register_key('Metrics/TestCostMax', window_length=1)
        self._logger.register_key('Metrics/TestM_cur', window_length=1)

        self._logger.register_key('Mics/RewUpdate')
        self._logger.register_key('Mics/CostUpdate')
        self._logger.register_key('Mics/CostIncreaseUpdate')
        self._logger.register_key('Mics/tolerance')
        self._logger.register_key('Loss/loss_vae')


    def _init_model(self) -> None:
        super()._init_model()

        self._cost_critic = (
            CriticBuilder_AUG(
                obs_space=self._env.observation_space,
                act_space=self._env.action_space,
                hidden_sizes=self._cfgs.model_cfgs.critic.hidden_sizes,
                activation=self._cfgs.model_cfgs.critic.activation,
                weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
                num_critics=2,
            )
            .build_critic('q')
            .to(self._device)
        )
        self._target_cost_critic = deepcopy(self._cost_critic)
        assert isinstance(
            self._cfgs.model_cfgs.critic.lr,
            float,
        ), 'The learning rate must be a float number.'
        self._cost_critic_optimizer = optim.Adam(
            self._cost_critic.parameters(),
            lr=self._cfgs.model_cfgs.critic.lr,
        )
        
        self._cost_increase_critic = (
            CriticBuilder_AUG(
                obs_space=self._env.observation_space,
                act_space=self._env.action_space,
                hidden_sizes=self._cfgs.model_cfgs.critic.hidden_sizes,
                activation=self._cfgs.model_cfgs.critic.activation,
                weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
                num_critics=2,
            )
            .build_critic('q')
            .to(self._device)
        )
        self._target_cost_increase_critic = deepcopy(self._cost_increase_critic)
        assert isinstance(
            self._cfgs.model_cfgs.critic.lr,
            float,
        ), 'The learning rate must be a float number.'
        self._cost_increase_critic_optimizer = optim.Adam(
            self._cost_increase_critic.parameters(),
            lr=self._cfgs.model_cfgs.critic.lr,
        )

        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        self.log_alpha_prime = nn.Parameter(torch.tensor(0.0), requires_grad = True)
        self.alpha_prime_optimizer = torch.optim.Adam([self.log_alpha_prime,], lr=self._cfgs.model_cfgs.critic.lr)

        self._vae = VAE_CAUG(
            state_dim = self._env.observation_space.shape[0]+1,     # Augment the observation space
            action_dim = self._env.action_space.shape[0],
            vae_features = self._cfgs.model_cfgs.vae['vae_features'],
            vae_layers = self._cfgs.model_cfgs.vae['vae_layers'],
            max_action = self._env.action_space.high[0]
        ).to(self._device)
        self._vae_optim = torch.optim.Adam(self._vae.parameters(), lr = self._cfgs.model_cfgs.vae['vae_lr'])

    # get the action and log_p of the obs and action
    def _get_policy_actions(self, obs, num_actions, network=None):
        '''
        Get the samples actions and probaility of the policy network.
        '''
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        with torch.no_grad():
            actions = network.predict(obs_temp, deterministic=True)
            logp_a = network.log_prob(actions)
        return actions, logp_a.view(obs.shape[0],num_actions,1)
    
    # get the value of the obs and action
    def _get_tensor_values(self, obs, actions, network=None):
        '''
        Get the Q-value 
        '''
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        q1_value_r, q2_value_r = network(obs_temp, actions)
        preds1 = q1_value_r.view(obs.shape[0], num_repeat, 1)
        preds2 = q2_value_r.view(obs.shape[0], num_repeat, 1)     
        return preds1, preds2

    def _train(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> None:
        obs, action, reward, cost, next_obs, done, cost_increase = batch
        reward = reward.squeeze()
        cost = cost.squeeze()
        done = done.squeeze()
        cost_increase = cost_increase.squeeze()

        obs= obs.to(torch.float32)
        next_obs = next_obs.to(torch.float32)
        cost_increase = cost_increase.to(torch.float32)

        self._update_vae(obs, action)
        # self._update_reward_critic(obs, action, reward, next_obs, done)
        self._update_reward_critic_cql(obs, action, reward, next_obs, done)
        # self._update_cost_critic(obs, action, cost, next_obs, done)
        # self._update_cost_increase_critic(obs, action, cost_increase, next_obs, done)
        self._update_cost_critic_con(obs, action, cost, next_obs, done)
        self._update_cost_increase_critic_con(obs, action, cost_increase, next_obs, done)
        self._update_actor_aug(obs, action)

        self._polyak_update()


    def _update_reward_critic_cql(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._actor.predict(next_obs, deterministic=False)
            next_logp = self._actor.log_prob(next_action)
            next_q1_value_r, next_q2_value_r = self._target_reward_critic(next_obs, next_action)
            next_q_value_r = torch.min(next_q1_value_r, next_q2_value_r) - next_logp * self._alpha
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        q1_value_r, q2_value_r = self._reward_critic.forward(obs, action)

        loss_q1 = ((target_q_value_r - q1_value_r)**2).mean()
        loss_q2 = ((target_q_value_r - q2_value_r)**2).mean()

        # add CQL
        random_actions_tensor = torch.FloatTensor(q1_value_r.shape[0]*self._num_random, action.shape[-1]).uniform_(-1,1).to(self._device)
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self._num_random, network=self._actor)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self._num_random, network=self._actor)
        q1_rand, q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self._reward_critic)
        q1_curr_actions, q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self._reward_critic)
        q1_next_actions, q2_next_actions = self._get_tensor_values(obs,new_curr_actions_tensor,network=self._reward_critic)

        # importance sammpled version
        random_density = torch.log(torch.tensor(0.5 ** curr_actions_tensor.shape[-1]))
        cat_q1 = torch.cat(
            [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
        )
        cat_q2 = torch.cat(
            [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
        )

        min_qf1_loss = torch.logsumexp(cat_q1 / self._temp, dim=1,).mean() * self._min_q_weight * self._temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self._temp, dim=1,).mean() * self._min_q_weight * self._temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - next_q1_value_r.mean() * self._min_q_weight
        min_qf2_loss = min_qf2_loss - next_q1_value_r.mean() * self._min_q_weight

        # with_lagrange
        alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
        min_qf1_loss = alpha_prime * (min_qf1_loss - self._target_action_gap)
        min_qf2_loss = alpha_prime * (min_qf2_loss - self._target_action_gap)


        # Update the alpha
        self.alpha_prime_optimizer.zero_grad()
        alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5
        alpha_prime_loss.backward(retain_graph=True)
        self.alpha_prime_optimizer.step()

        qf1_loss = loss_q1 + min_qf1_loss
        qf2_loss = loss_q2 + min_qf2_loss
 
        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._reward_critic.critic_0.parameters():
                qf1_loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

            for param in self._reward_critic.critic_1.parameters():
                qf2_loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff


        # Update q1
        self._reward_critic_optimizer1.zero_grad()
        qf1_loss.backward(retain_graph=True)

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._reward_critic.critic_0.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._reward_critic_optimizer1.step()

        # Update q2
        self._reward_critic_optimizer2.zero_grad()
        qf2_loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._reward_critic.critic_1.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._reward_critic_optimizer2.step()

        self._logger.store(
            **{
                'Loss/Loss_reward_critic': qf1_loss.item(),
                'Qr/data_Qr': q1_value_r.mean().item(),
                'Qr/target_Qr': target_q_value_r.mean().item(),
            },
        )

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._actor.predict(next_obs, deterministic=False)
            qc1_target, qc2_target = self._target_cost_critic(next_obs, next_action)
            qc_target = torch.min(qc1_target, qc2_target)
            qc_target = cost + (1 - done) * self._cfgs.algo_cfgs.gamma * qc_target.unsqueeze(1)
            qc_target = qc_target.squeeze(1)

        qc1, qc2 = self._cost_critic.forward(obs, action)
        cost_critic_loss = nn.functional.mse_loss(qc1, qc_target) + nn.functional.mse_loss(
            qc2,
            qc_target,
        )
        self._cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self._cost_critic_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_cost_critic': cost_critic_loss.item(),
                'Qc/data_Qc': qc1[0].mean().item(),
                'Qc/target_Qc': qc_target[0].mean().item(),
            },
        )

    # Contain the condition bellman equation
    def _update_cost_critic_con(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Update cost critic.

        - Get the TD loss of cost critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor.predict(next_obs, deterministic=True)
            next_qc_value_c = self._target_cost_critic(next_obs, next_action)[0]
            target_qc_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_qc_value_c
        qc_value_c = self._cost_critic(obs, action)[0]

        # The current Qc
        obs_repeat = obs.repeat((self._cfgs.algo_cfgs.action_num,1,1)).reshape(-1, obs.shape[-1])
        next_obs_repeat = next_obs.repeat((self._cfgs.algo_cfgs.action_num,1,1)).reshape(-1, next_obs.shape[-1])
        qc_ood_curr_act = self._actor.predict(obs_repeat, deterministic=False)
        qc_ood_next_act = self._actor.predict(next_obs_repeat, deterministic=False)
        qc_ood_curr_pred = self._cost_critic(obs_repeat, qc_ood_curr_act)[0]
        qc_ood_next_pred = self._cost_critic(next_obs_repeat, qc_ood_next_act)[0]

        qc_ood_pred = torch.cat([qc_ood_curr_pred, qc_ood_next_pred],0)
        qc_ood_curr_act = qc_ood_curr_act.view(self._cfgs.algo_cfgs.action_num, self._cfgs.algo_cfgs.batch_size, -1)
        qc_ood_next_act = qc_ood_next_act.view(self._cfgs.algo_cfgs.action_num, self._cfgs.algo_cfgs.batch_size, -1)

        # The pesudo Qc
        qc_curr_act = self._vae.decode_multiple(obs, num = self._cfgs.algo_cfgs.action_num)
        qc_next_act = self._vae.decode_multiple(next_obs, num = self._cfgs.algo_cfgs.action_num)
        qc_curr_act = qc_curr_act.reshape(self._cfgs.algo_cfgs.action_num * self._cfgs.algo_cfgs.batch_size, -1)
        qc_next_act = qc_next_act.reshape(self._cfgs.algo_cfgs.action_num * self._cfgs.algo_cfgs.batch_size, -1)
        pesudo_qc_curr_target = self._cost_critic(obs_repeat, qc_curr_act)[0]
        pesudo_qc_next_target = self._cost_critic(next_obs_repeat, qc_next_act)[0]

        pesudo_qc_target = torch.cat([pesudo_qc_curr_target, pesudo_qc_next_target],0)
        qc_curr_act = qc_curr_act.view(self._cfgs.algo_cfgs.action_num, self._cfgs.algo_cfgs.batch_size, -1)
        qc_next_act = qc_next_act.view(self._cfgs.algo_cfgs.action_num, self._cfgs.algo_cfgs.batch_size, -1)
        pesudo_qc_target = pesudo_qc_target.detach()

        assert pesudo_qc_target.shape[0] == qc_ood_pred.shape[0]

        qc_deviation = qc_ood_pred - pesudo_qc_target
        # qc_deviation[qc_deviation<=0]=0
        qc_ood_loss = torch.mean(qc_deviation**2)
        qc_loss = self._cfgs.algo_cfgs.lam * nn.functional.mse_loss(qc_value_c, target_qc_value_c) + (1-self._cfgs.algo_cfgs.lam) * qc_ood_loss

        # loss = nn.functional.mse_loss(qc_value_c, target_q_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._cost_critic.parameters():
                qc_loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        # Update the qc network
        self._cost_critic_optimizer.zero_grad()
        qc_loss.backward()
        # if self._cfgs.algo_cfgs.max_grad_norm:
        #     clip_grad_norm_(
        #         self._cost_critic.parameters(),
        #         self._cfgs.algo_cfgs.max_grad_norm,
        #     )
        self._cost_critic_optimizer.step()


        self._logger.store(
            **{
                'Loss/Loss_cost_critic': qc_loss.item(),
                'Qc/data_Qc': qc_value_c.mean().item(),
                'Qc/target_Qc': target_qc_value_c.mean().item(),
            },
        )

    def _update_cost_increase_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost_increase: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._actor.predict(next_obs, deterministic=False)
            qc1_target_increase, qc2_target_increase = self._target_cost_increase_critic(next_obs, next_action)
            qc_target_increase = torch.min(qc1_target_increase, qc2_target_increase)
            qc_target_increase = cost_increase + (1 - done) * qc_target_increase.unsqueeze(1)
            qc_target_increase = qc_target_increase.squeeze(1)

        qc1_increase, qc2_increase = self._cost_increase_critic.forward(obs, action)
        cost_increase_critic_loss = nn.functional.mse_loss(qc1_increase, qc_target_increase) + nn.functional.mse_loss(
            qc2_increase,
            qc_target_increase,
        )
        self._cost_increase_critic_optimizer.zero_grad()
        cost_increase_critic_loss.backward()
        self._cost_increase_critic_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_cost_critic_increase': cost_increase_critic_loss.item(),
                'Qc/data_Qc_increase': qc1_increase[0].mean().item(),
                'Qc/target_Qc_increase': qc_target_increase[0].mean().item(),
            },
        )
    
    # Contain the conditional bellman equation
    def _update_cost_increase_critic_con(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost_increase: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Update cost increase critic.

        - Get the TD loss of cost increase critic without the .
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
            cost_increase (torch.Tensor): The ``cost increase`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor.predict(next_obs, deterministic=True)
            next_qd_value_c = self._target_cost_increase_critic(next_obs, next_action)[0]
            target_qd_value_c = cost_increase + (1 - done) * next_qd_value_c
        qd_value_c = self._cost_increase_critic(obs, action)[0]

        # The current Qd
        obs_repeat = obs.repeat((self._cfgs.algo_cfgs.action_num,1,1)).reshape(-1, obs.shape[-1])
        next_obs_repeat = next_obs.repeat((self._cfgs.algo_cfgs.action_num,1,1)).reshape(-1, next_obs.shape[-1])
        qd_ood_curr_act = self._actor.predict(obs_repeat, deterministic=False)
        qd_ood_next_act = self._actor.predict(next_obs_repeat, deterministic=False)
        qd_ood_curr_pred = self._cost_increase_critic(obs_repeat, qd_ood_curr_act)[0]
        qd_ood_next_pred = self._cost_increase_critic(next_obs_repeat, qd_ood_next_act)[0]

        qd_ood_pred = torch.cat([qd_ood_curr_pred, qd_ood_next_pred],0)
        qd_ood_curr_act = qd_ood_curr_act.view(self._cfgs.algo_cfgs.action_num, self._cfgs.algo_cfgs.batch_size, -1)
        qd_ood_next_act = qd_ood_next_act.view(self._cfgs.algo_cfgs.action_num, self._cfgs.algo_cfgs.batch_size, -1)

        # The pesudo Qd
        qd_curr_act = self._vae.decode_multiple(obs, num = self._cfgs.algo_cfgs.action_num)
        qd_next_act = self._vae.decode_multiple(next_obs, num = self._cfgs.algo_cfgs.action_num)
        qd_curr_act = qd_curr_act.reshape(self._cfgs.algo_cfgs.action_num * self._cfgs.algo_cfgs.batch_size, -1)
        qd_next_act = qd_next_act.reshape(self._cfgs.algo_cfgs.action_num * self._cfgs.algo_cfgs.batch_size, -1)
        pesudo_qd_curr_target = self._cost_increase_critic(obs_repeat, qd_curr_act)[0]
        pesudo_qd_next_target = self._cost_increase_critic(next_obs_repeat, qd_next_act)[0]

        pesudo_qd_target = torch.cat([pesudo_qd_curr_target, pesudo_qd_next_target],0)
        qd_curr_act = qd_curr_act.view(self._cfgs.algo_cfgs.action_num, self._cfgs.algo_cfgs.batch_size, -1)
        qd_next_act = qd_next_act.view(self._cfgs.algo_cfgs.action_num, self._cfgs.algo_cfgs.batch_size, -1)
        pesudo_qd_target = pesudo_qd_target.detach()

        assert pesudo_qd_target.shape[0] == qd_ood_pred.shape[0]

        qd_deviation = qd_ood_pred - pesudo_qd_target
        # qd_deviation[qd_deviation<=0]=0
        # qd_deviation[qd_deviation>=0]=0
        qd_ood_loss = torch.mean(qd_deviation**2)
        qd_loss = self._cfgs.algo_cfgs.lam * nn.functional.mse_loss(qd_value_c, target_qd_value_c) + (1-self._cfgs.algo_cfgs.lam) * qd_ood_loss


        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_increase_critic.parameters():
                qd_loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        # Update the qd network
        self._cost_increase_critic_optimizer.zero_grad()
        qd_loss.backward()
        # if self._cfgs.algo_cfgs.max_grad_norm:
        #     clip_grad_norm_(
        #         self._cost_increase_critic.parameters(),
        #         self._cfgs.algo_cfgs.max_grad_norm,
        #     )
        self._cost_increase_critic_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_cost_critic_increase': qd_loss.item(),
                'Qc/data_Qc_increase': qd_value_c.mean().item(),
                'Qc/target_Qc_increase': target_qd_value_c.mean().item(),
            },
        )

    def _update_actor_aug(  # pylint: disable=too-many-locals
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        # current Q-value of the reward
        qr1, qr2 = self._reward_critic.forward(obs, action)
        qr_data = torch.min(qr1, qr2)

        # current Q-value of the cost
        qc1, qc2 = self._cost_critic.forward(obs, action)
        qc_data = torch.min(qc1, qc2)

        # current Q-value of the cost increase
        qc1_increase, qc2_increase = self._cost_increase_critic.forward(obs, action)
        qc_increase_data = torch.min(qc1_increase, qc2_increase)

        obs_repeat = (
            obs.unsqueeze(1)
            .repeat(1, self._cfgs.algo_cfgs.sampled_action_num, 1)
            .view(obs.shape[0] * self._cfgs.algo_cfgs.sampled_action_num, obs.shape[1])
        )
        act_sample = self._actor.predict(obs_repeat, deterministic=False)

        # The advantage for the reward Q-value
        qr1_sample, qr2_sample = self._reward_critic.forward(obs_repeat, act_sample)
        qr_sample = torch.min(qr1_sample, qr2_sample)
        mean_qr = torch.vstack(
            [q.mean() for q in qr_sample.reshape(-1, self._cfgs.algo_cfgs.sampled_action_num, 1)],
        )
        adv_r = qr_data - mean_qr.squeeze(1)

        # The advantage for the cost Q-value
        qc1_sample, qc2_sample = self._cost_critic.forward(obs_repeat, act_sample)
        qc_sample = torch.min(qc1_sample, qc2_sample)
        mean_qc = torch.vstack(
            [q.mean() for q in qc_sample.reshape(-1, self._cfgs.algo_cfgs.sampled_action_num, 1)],
        )
        adv_c = qc_data - mean_qc.squeeze(1)
        
        # The advantage for the cost increase Q-value
        qc1_increase_sample, qc2_increase_sample = self._cost_increase_critic.forward(obs_repeat, act_sample)
        qc_increase_sample = torch.min(qc1_increase_sample, qc2_increase_sample)
        mean_qc_increase = torch.vstack(
            [q.mean() for q in qc_increase_sample.reshape(-1, self._cfgs.algo_cfgs.sampled_action_num, 1)],
        )
        adv_c_increase = qc_increase_data - mean_qc_increase.squeeze(1)

        # Set the actor loss based on the crpo
        if self._step > 0.00* self._cfgs.train_cfgs.total_steps:
            loss_c_value = self._logger.get_stats('Metrics/EpCost')[0]
            loss_c_increase_value =  self._logger.get_stats('Metrics/TestCostMax')[0]
            loss_r_value =  self._logger.get_stats('Metrics/EpRet')[0]
            if (loss_c_value > self._cfgs.algo_cfgs.cost_limit + self._cfgs.algo_cfgs.tolerance and loss_r_value > self._cfgs.algo_cfgs.r_value):
                self._cost_update += 1
                loss_adv = - adv_c
            elif loss_c_increase_value > self._cfgs.algo_cfgs.cost_increase_limit and loss_r_value > self._cfgs.algo_cfgs.r_value:
                self._cost_increase_update += 1
                loss_adv = - adv_c_increase
            else: 
                self._rew_update += 1
                loss_adv = adv_r
        else:
            self._rew_update += 1
            loss_adv = adv_r

        exp_adv = torch.exp(loss_adv.detach() / self._cfgs.algo_cfgs.beta)
        exp_adv = torch.clamp(exp_adv, 0, 1e10)

        self._actor(obs)
        logp = self._actor.log_prob(action)
        bc_loss = -logp
        policy_loss = (exp_adv * bc_loss).mean()
        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._actor_optimizer.step()


        self._logger.store(
            **{
                'Loss/Loss_actor': policy_loss.item(),
                'Qr/current_Qr': qr_data[0].mean().item(),
                'Qc/current_Qc': qc_data[0].mean().item(),
                'Qc/current_Qc_increase':qc_increase_data[0].mean().item(),
                'Train/PolicyStd': self._actor.std,
                'Mics/RewUpdate': self._rew_update,
                'Mics/CostUpdate': self._cost_update,
                'Mics/CostIncreaseUpdate': self._cost_increase_update,
                'Mics/tolerance': self._cfgs.algo_cfgs.cost_limit + self._cfgs.algo_cfgs.tolerance,
            },
        )

    def _update_actor(  # pylint: disable=too-many-locals
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        qr1, qr2 = self._reward_critic.forward(obs, action)
        qr_data = torch.min(qr1, qr2)

        qc1, qc2 = self._cost_critic.forward(obs, action)
        qc_data = torch.min(qc1, qc2)

        obs_repeat = (
            obs.unsqueeze(1)
            .repeat(1, self._cfgs.algo_cfgs.sampled_action_num, 1)
            .view(obs.shape[0] * self._cfgs.algo_cfgs.sampled_action_num, obs.shape[1])
        )
        act_sample = self._actor.predict(obs_repeat, deterministic=False)

        qr1_sample, qr2_sample = self._reward_critic.forward(obs_repeat, act_sample)
        qr_sample = torch.min(qr1_sample, qr2_sample)
        mean_qr = torch.vstack(
            [q.mean() for q in qr_sample.reshape(-1, self._cfgs.algo_cfgs.sampled_action_num, 1)],
        )
        adv_r = qr_data - mean_qr.squeeze(1)

        qc1_sample, qc2_sample = self._reward_critic.forward(obs_repeat, act_sample)
        qc_sample = torch.min(qc1_sample, qc2_sample)
        mean_qc = torch.vstack(
            [q.mean() for q in qc_sample.reshape(-1, self._cfgs.algo_cfgs.sampled_action_num, 1)],
        )
        adv_c = qc_data - mean_qc.squeeze(1)

        exp_adv = torch.exp(
            (adv_r - self._lagrange.lagrangian_multiplier.item() * adv_c).detach()
            / self._cfgs.algo_cfgs.beta,
        )
        exp_adv = torch.clamp(exp_adv, 0, 1e10)

        self._actor(obs)
        logp = self._actor.log_prob(action)
        bc_loss = -logp
        policy_loss = (exp_adv * bc_loss).mean()
        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._actor_optimizer.step()

        if (
            self.epoch * self._cfgs.algo_cfgs.steps_per_epoch
            > self._cfgs.algo_cfgs.lagrange_start_step
        ):
            self._lagrange.update_lagrange_multiplier(mean_qc.mean().item())

        self._logger.store(
            **{
                'Loss/Loss_actor': policy_loss.item(),
                'Qr/current_Qr': qr_data[0].mean().item(),
                'Qc/current_Qc': qc_data[0].mean().item(),
                'Train/PolicyStd': self._actor.std,
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            },
        )

    # Update the vae Network
    def _update_vae(self, obs, action):

        dist, _action = self._vae(obs, action)
        kl_loss = torch.distributions.kl.kl_divergence(dist, Normal(0, 1)).mean()
        recon_loss = nn.functional.mse_loss(action, _action)
        vae_loss = kl_loss + recon_loss

        self._vae_optim.zero_grad()
        vae_loss.backward()
        self._vae_optim.step()

        self._logger.store(
            {
                'Loss/loss_vae': vae_loss.mean().item(),
            },
        )

    def _polyak_update(self) -> None:
        super()._polyak_update()
        for target_param, param in zip(
            self._target_cost_critic.parameters(),
            self._cost_critic.parameters(),                     
        ):
            target_param.data.copy_(
                self._cfgs.algo_cfgs.polyak * param.data
                + (1 - self._cfgs.algo_cfgs.polyak) * target_param.data,
            )

        for target_param, param in zip(
            self._target_cost_increase_critic.parameters(),
            self._cost_increase_critic.parameters(),                     
        ):
            target_param.data.copy_(
                self._cfgs.algo_cfgs.polyak * param.data
                + (1 - self._cfgs.algo_cfgs.polyak) * target_param.data,
            )
