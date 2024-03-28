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
"""This module contains the model for all methods."""

from omnisafe.models.actor import ActorBuilder
from omnisafe.models.actor.gaussian_actor import GaussianActor
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor
from omnisafe.models.actor.mlp_actor import MLPActor
from omnisafe.models.actor.perturbation_actor import PerturbationActor
from omnisafe.models.actor.vae_actor import VAE
from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.actor_critic.actor_q_critic import ActorQCritic
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.base import Actor, Critic
from omnisafe.models.critic import CriticBuilder
from omnisafe.models.critic.q_critic import QCritic
from omnisafe.models.critic.v_critic import VCritic
from omnisafe.models.offline.dice import ObsEncoder


from omnisafe.models.base_aug import Actor_AUG, Critic_AUG
