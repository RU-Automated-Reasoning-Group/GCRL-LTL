from stable_baselines3.common.callbacks import BaseCallback


class CollectTrajectoryCallback(BaseCallback):

    def __init__(self, traj_buffer, verbose: int = 0):
        super().__init__(verbose)
        self.traj_buffer = traj_buffer

    def _on_rollout_end(self) -> None:
        self.traj_buffer.add_rollouts(self.model.rollout_buffer)

    def _on_step(self) -> bool:
        return True
