from typing import Optional, Type, List, Dict, Any

import numpy as np
import torch as th
import torch.nn as nn
import gym
from gym import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.torch_layers import create_mlp


class GCVNetwork(BasePolicy):
    """
    Goal-Conditioned-Value (GC-Value) network for GC-PPO

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super(GCVNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        action_dim = self.action_space.n  # number of actions
                                          # action_dim doesn't matter
        gcvf_net = create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        self.gcvf_net = nn.Sequential(*gcvf_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the GC-Values.

        :param obs: Observation
        :return: The estimated GC-Value for each state.
        """
        return self.gcvf_net(self.extract_features(obs))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        gc_value = self.forward(observation).reshape(-1)
        return gc_value

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class GCPPO(PPO):

    ZONE_OBS_DIM = 24

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gcvf = self.make_gcvf()
        self.gcvf_optimizer = th.optim.Adam(self.gcvf.parameters(), lr=kwargs['learning_rate'])
        
    def make_gcvf(self) -> GCVNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=self.policy.features_extractor,
            features_dim=self.policy.features_dim,
            net_arch=[512, 1024, 256],
        )
        return GCVNetwork(**net_args).to(self.device)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, gc_value_losses = [], [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # set start information
                observations = rollout_data.observations
                observations = observations[:, :self.ZONE_OBS_DIM] = 0.5
                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Get current GC-values estimates
                full_obseravtions = rollout_data.observations
                gc_values_pred = self.gcvf(full_obseravtions).flatten()
                gc_value_loss = F.mse_loss(rollout_data.returns, gc_values_pred)
                gc_value_losses.append(gc_value_loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                self.gcvf_optimizer.zero_grad()
                loss.backward()
                gc_value_loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                th.nn.utils.clip_grad_norm_(self.gcvf.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.gcvf_optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/gc_value_loss", np.mean(gc_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
