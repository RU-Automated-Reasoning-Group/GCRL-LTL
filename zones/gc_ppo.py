from typing import NamedTuple, Optional, Generator
import copy

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class GCRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    rewards: th.Tensor


class GCRolloutBuffer(RolloutBuffer):
    def get(self, batch_size: Optional[int] = None) -> Generator[GCRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "rewards",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> GCRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.rewards[batch_inds].flatten(),
        )
        return GCRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class GCPPO(PPO):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_net = self.make_q_net()
        self.non_terminal_clock = 0
        self.qf_coef = 0.5  # vf_coef = 0.5, policy_coef = 1.0
        
    def make_q_net(self) -> QNetwork:
        # NOTE: note from DQN, what does it mean?
        # Make sure we always have separate networks for features extractors etc
        net_args = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=self.policy.features_extractor,
            features_dim=self.policy.features_dim,
            net_arch=[256, 256],
        )
        return QNetwork(**net_args).to(self.device)
    
    def _setup_model(self) -> None:
        super()._setup_model()
        buffer_cls = GCRolloutBuffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

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
        pg_losses, value_losses = [], []
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

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
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

                # Get current Q-values estimates
                current_q_values = self.q_net(rollout_data.observations)
                #print(current_q_values, current_q_values.shape, type(current_q_values))  # (20, 4)

                # Retrieve the q-values for the actions from the replay buffer
                current_q_values = th.gather(current_q_values, dim=1, index=rollout_data.actions.long())
                #print(current_q_values, current_q_values.shape, type(current_q_values)) # (20, 1)

                # sparse rewards can infer dones
                dones = rollout_data.rewards
                dones_list = dones.numpy().tolist()
                rev_dones_list = dones.numpy().tolist()
                rev_dones_list.reverse()
                first_done_idx = dones_list.index(1) if 1 in dones_list else self.batch_size
                last_done_idx = self.batch_size - rev_dones_list.index(1) if 1 in rev_dones_list else 1000
                
                if first_done_idx + self.non_terminal_clock > self.batch_size:
                    force_done_idx = self.batch_size - self.non_terminal_clock
                    dones[force_done_idx] = 1
                    last_done_idx = max(force_done_idx, last_done_idx)
                self.non_terminal_clock = self.batch_size - last_done_idx
                
                target_q_values = rollout_data.rewards + (1 - dones) * self.gamma * values_pred
                q_value_loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(0))

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.qf_coef * q_value_loss

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
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
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
