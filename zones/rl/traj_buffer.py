from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
from gym import spaces


class TrajectoryBuffer:
    """
    Trajectory buffer used to train Goal-Value function.


    :param traj_length: Max number of RolloutBuffers in one continual trajectory
    :param buffer_size: Max number of continual trajectories

    total number of stored trajectories = traj_length * buffer_size 
    """
    def __init__(
        self, 
        traj_length: int = 4,
        buffer_size: int = 10000,
        device: Union[torch.device, str] = "cpu", 
        n_envs: int = 1,
    ):
        self.traj_length = traj_length
        self.buffer_size = buffer_size
        self.device = device
        self.n_envs = n_envs
        self.reset()

    def reset(self) -> None:
        self.rollouts = np.zeros((self.buffer_size * self.traj_length, self.n_envs), dtype=np.float32)
    
    def add_rollout(self, ) -> None:
        pass

    def add(self, ) -> None:
        pass
