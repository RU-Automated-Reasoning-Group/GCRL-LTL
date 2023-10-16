"""Helper functions to create envs easily through one interface.

- create_env(env_name)
- get_goal_threshold(env_name)
"""

import numpy as np
from algo.envs.env_utils import DiscretizedActionEnv

try:
    import mujoco_py
except:
    print('MuJoCo must be installed.')


from algo.envs.ant_16rooms import Ant16RoomsEnv
env_names = ['ant16rooms','ant16roomscloseddoors']


def create_env(env_name):
    """Helper function."""
    assert env_name in env_names
    if env_name == 'ant16rooms':
        return Ant16RoomsEnv(silent=True)
    else:
        raise NotImplementedError()


def get_env_params(env_name, images=False):
    assert env_name in env_names

    base_params = dict(
        eval_freq=10000,
        eval_episodes=50,
        max_trajectory_length=50,
        max_timesteps=1e6,
    )
    # ant envs basic setting:
    # env_specific_params = dict(
    #     goal_threshold=1,
    #     max_trajectory_length=1000,
    #     max_timesteps=2e5,
    #     eval_episodes=2,
    #     action_granularity=5,
    # )
    if env_name == 'ant16rooms':
        env_specific_params = dict(
            goal_threshold=1,
            max_trajectory_length=1000,
            max_timesteps=2e4,
            eval_episodes=2,
            action_granularity=2,
            graph_node_cover_size=1,
        )
    else:
        raise NotImplementedError()

    base_params.update(env_specific_params)
    return base_params
