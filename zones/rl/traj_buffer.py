from typing import Any

import torch
from torch.utils.data import Dataset
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer

from envs.utils import get_zone_vector


class TrajectoryBufferDataset(Dataset):
    def __init__(self, states, goal_values) -> None:
        self.states = states
        self.goal_values = goal_values
        
    def __getitem__(self, index) -> Any:
        s = self.states[index]
        v_omega = self.goal_values[index]
        return s, v_omega

    def __len__(self) -> int:
        return len(self.states)


class TrajectoryBuffer:

    ZONE_OBS_DIM = 24

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
        self.zone_vector = get_zone_vector()
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

    def build_dataset(self, policy):
        states, goal_values = [], []
        buffer_length = int(self.buffer_size / self.n_envs)
        with torch.no_grad():
            for pid in range(self.n_envs):
                local_states, local_goal_values = [], []
                pos, forward_steps = 0, 0
                while pos < buffer_length:
                    local_states.append(self.buffers[pid]['obs'][pos])
                    if forward_steps >= self.traj_length - 1 and not self.buffers[pid]['success']:
                        local_states, local_goal_values = [], []
                        forward_steps = 0
                    elif self.buffers[pid]['success']:
                        # TODO: compute the goal-value for all possible goals
                        for state in local_states:
                            local_goal_values.append(self.get_goal_value(state), policy)
                        states += local_states
                        goal_values += local_goal_values
                        local_states, local_goal_values = [], []
                        forward_steps = 0
                    
                    pos += 1
                    forward_steps += 1

        return TrajectoryBufferDataset(states=state, goal_values=goal_values)

    def get_goal_value(self, state, policy):
        goal_value = -np.ones((4), dtype=np.float32)
        for idx, zone in enumerate(self.zone_vector):
            if not np.array_equal(state[-self.ZONE_OBS_DIM:], self.zone_vector[zone]):
                with torch.no_grad():
                    goal_value[idx] = policy.predict_value(np.concatenate(state, self.zone_vector[zone]))
        
        return goal_value
