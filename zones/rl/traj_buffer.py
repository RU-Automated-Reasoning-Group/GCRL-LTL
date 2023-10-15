from typing import Union

import numpy as np
import torch

from stable_baselines3.common.buffers import RolloutBuffer


class TrajectoryBuffer:
    """
    Trajectory buffer used to train Goal-Value function.


    :param traj_length: Max number of RolloutBuffers in one continual trajectory
    :param buffer_size: Max number of continual trajectories

    total number of stored trajectories = traj_length * buffer_size 
    """
    def __init__(
        self, 
        buffer_size: int = 10000,
        device: Union[torch.device, str] = "cpu", 
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.n_envs = n_envs
        self.reset()

    def reset(self) -> None:
        self.rollouts = dict(zip(range(self.n_envs), [[] for _ in range(self.n_envs)]))
    
    def add_rollout(self, rollout_buffer: RolloutBuffer) -> None:
        pass

    def revoke_rollout(self) -> None:
        pass

    def push_rollout(self) -> None:
        pass

    def add(self, ) -> None:
        pass
