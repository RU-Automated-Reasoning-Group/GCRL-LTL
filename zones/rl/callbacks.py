import os

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization


class CollectTrajectoryCallback(BaseCallback):

    def __init__(self, traj_buffer, verbose: int = 0):
        super().__init__(verbose)
        self.traj_buffer = traj_buffer

    def _on_rollout_start(self) -> None:
        pass
    
    def _on_rollout_end(self) -> None:
        # NOTE: under construction
        self.traj_buffer.add_rollouts(self.model.rollout_buffer)

    def _on_step(self) -> bool:
        return True
