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
"""Implementation of MLPActorActor."""

from __future__ import annotations

import torch
from torch.distributions import Distribution

from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


# pylint: disable-next=too-many-instance-attributes
class MLPActor(Actor):
    """Implementation of MLPActor.

    MLPActor is a Gaussian actor with a learnable mean value. It is used in off-policy algorithms
    such as ``DDPG``, ``TD3`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        output_activation (Activation, optional): Output activation function. Defaults to ``'tanh'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        output_activation: Activation = 'tanh',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`MLPActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self.net: torch.nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim],
            activation=activation,
            output_activation=output_activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self._noise: float = 0.1

    def predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Predict the action given observation.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the mean of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to True.
        """
        action = self.net(obs)
        if deterministic:
            return action
        with torch.no_grad():
            noise = torch.normal(0, self._noise * torch.ones_like(action))
            return torch.clamp(action + noise, -1, 1)

    @property
    def noise(self) -> float:
        """Noise of the action."""
        return self._noise

    @noise.setter
    def noise(self, noise: float) -> None:
        """Set the action noise."""
        assert noise >= 0, 'Noise should be non-negative.'
        self._noise = noise

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward method implementation.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The distribution of the action.
        """
        return self._distribution(obs)

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Log probability of the action.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward`  tensor.

        Raises:
            NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return self._noise
