from typing import Optional, Type, List, Dict, Any

import torch as th
import torch.nn as nn
import gym

from stable_baselines3.common.policies import BasePolicy
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
        normalize_images: bool = True,
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
        self.normalize_images = normalize_images

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
        gc_value = self.forward(observation)
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
