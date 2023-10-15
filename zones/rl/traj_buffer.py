from typing import Union

import numpy as np
import torch

from stable_baselines3.common.buffers import RolloutBuffer


class TrajectoryBuffer:
    def __init__(
        self, 
        traj_length: int = 1000,
        buffer_size: int = 100000,
        obs_dim: int = 100,
        n_envs: int = 1,
    ):
        self.traj_length = traj_length
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.n_envs = n_envs
        self.reset()
    
    def reset(self) -> None:
        self.buffers = {}
        buffer_length = int(self.buffer_size / self.n_envs)
        for pid in range(self.n_envs):
            self.buffers[pid] = {
                'obs': np.zeros((buffer_length, self.obs_dim), dtype=np.float32),
                'success': np.zeros((buffer_length, 1), dtype=np.bool), 
                'steps': np.zeros((buffer_length, 1), dtype=np.int),
            }
        self.pos = 0

    def add_rollouts(self, rollout_buffer: RolloutBuffer) -> None:
        rollout_obs = rollout_buffer.observations['obs'].transpose(1, 0, 2)  # (n_envs, n_steps/n_envs, obs_dim)
        rollout_success = rollout_buffer.observations['success'].transpose(1, 0, 2)
        rollout_steps = rollout_buffer.observations['steps'].transpose(1, 0, 2)
        
        shape = rollout_obs.shape
        forward_steps = shape[1]

        for pid in range(self.n_envs):
            self.buffers[pid]['obs'][self.pos: self.pos + forward_steps] = rollout_obs[pid]
            self.buffers[pid]['success'][self.pos: self.pos + forward_steps] = rollout_success[pid]
            self.buffers[pid]['steps'][self.pos: self.pos + forward_steps] = rollout_steps[pid]

        self.pos += forward_steps
        
        print(self.pos)
        print(self.buffers[0]['steps'][-500: -1])  # NOTE: debug this
        exit()
