"""Helper functions to create envs easily through one interface.

- create_env(env_name)
- get_goal_threshold(env_name)
"""

import numpy as np
from gcsl.envs.env_utils import DiscretizedActionEnv

try:
    import mujoco_py
except:
    print('MuJoCo must be installed.')

from gcsl.envs.room_env import PointmassGoalEnv
from gcsl.envs.sawyer_push import SawyerPushGoalEnv
from gcsl.envs.sawyer_door import SawyerDoorGoalEnv
from gcsl.envs.lunarlander import LunarEnv
from gcsl.envs.claw_env import ClawEnv
from gcsl.envs.ant_envs_in_gcsl import AntFallEnv, AntUMazeEnv, AntPushEnv
from gcsl.envs.ant_envs_in_hac import AntFourRoomsEnv
from gcsl.envs.ant_envs_in_ris import AntLongUMazeEnv, AntSMazeEnv, AntPiMazeEnv, AntOmegaMazeEnv
from gcsl.envs.ant_16rooms import Ant16RoomsEnv, Ant16RoomswithClosedDoorsEnv
from gcsl.envs.ant_9rooms import Ant9RoomsEnv, Ant9RoomswithClosedDoorsEnv
from gcsl.envs.FetchSlide import FetchSlideEnv
env_names = ['pointmass_rooms', 'pointmass_empty', 'pusher', 'lunar', 'door', 'claw', 'antfall', 'antumaze',
             'antpush', 'antfourrooms', 'antlongumaze', 'antsmaze', 'antpimaze', 'antomegamaze', 'ant16rooms', 'ant9rooms', 'ant9roomscloseddoors', 'ant16roomscloseddoors','fetchslide']


def create_env(env_name):
    """Helper function."""
    assert env_name in env_names
    if env_name == 'pusher':
        return SawyerPushGoalEnv()
    elif env_name == 'door':
        return SawyerDoorGoalEnv()
    elif env_name == 'pointmass_empty':
        return PointmassGoalEnv(room_type='empty')
    elif env_name == 'pointmass_rooms':
        return PointmassGoalEnv(room_type='rooms')
    elif env_name == 'lunar':
        return LunarEnv()
    elif env_name == 'claw':
        return ClawEnv()
    elif env_name == 'antfall':
        return AntFallEnv(silent=True)
    elif env_name == 'antumaze':
        return AntUMazeEnv(silent=True)
    elif env_name == 'antpush':
        return AntPushEnv(silent=True)
    elif env_name == 'antfourrooms':
        return AntFourRoomsEnv(silent=True)
    elif env_name == 'antlongumaze':
        return AntLongUMazeEnv(silent=True)
    elif env_name == 'antsmaze':
        return AntSMazeEnv(silent=True)
    elif env_name == 'antpimaze':
        return AntPiMazeEnv(silent=True)
    elif env_name == 'antomegamaze':
        return AntOmegaMazeEnv(silent=True)
    elif env_name == 'ant9rooms':
        return Ant9RoomsEnv(silent=True)
    elif env_name == 'ant9roomscloseddoors':
        return Ant9RoomswithClosedDoorsEnv(silent=True)
    elif env_name == 'ant16rooms':
        return Ant16RoomsEnv(silent=True)
    elif env_name == 'ant16roomscloseddoors':
        return Ant16RoomswithClosedDoorsEnv(silent=True)
    elif env_name == 'fetchslide':
        return FetchSlideEnv()


def get_env_params(env_name, images=False):
    assert env_name in env_names

    base_params = dict(
        eval_freq=10000,
        eval_episodes=50,
        max_trajectory_length=50,
        max_timesteps=1e6,
    )

    if env_name == 'pusher':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif 'pick' in env_name:
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'door':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif 'pointmass' in env_name:
        env_specific_params = dict(
            goal_threshold=0.08,
            max_timesteps=2e5,
            eval_freq=2000,
        )
    elif env_name == 'lunar':
        env_specific_params = dict(
            goal_threshold=0.08,
            max_timesteps=2e5,
            eval_freq=2000,
        )
    elif env_name == 'claw':
        env_specific_params = dict(
            goal_threshold=0.1,
        )
# ant envs basic setting:
# env_specific_params = dict(
#     goal_threshold=1,
#     max_trajectory_length=1000,
#     max_timesteps=2e5,
#     eval_episodes=2,
#     action_granularity=5,
# )
    elif env_name == 'antfall':
        env_specific_params = dict(
            goal_threshold=1,
            max_trajectory_length=1000,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=1,
        )
    elif env_name == 'antumaze':
        env_specific_params = dict(
            goal_threshold=1,
            max_trajectory_length=1000,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=1,
        )
    elif env_name == 'antpush':
        env_specific_params = dict(
            goal_threshold=1,
            max_trajectory_length=1000,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=1,
        )
    elif env_name == 'antfourrooms':
        env_specific_params = dict(
            goal_threshold=0.5,
            max_trajectory_length=1000,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=0.5,
        )
    elif env_name == 'antlongumaze':
        env_specific_params = dict(
            goal_threshold=0.5,
            max_trajectory_length=1000,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=0.5,
        )
    elif env_name == 'antsmaze':
        env_specific_params = dict(
            goal_threshold=0.5,
            max_trajectory_length=1000,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=0.5,
        )
    elif env_name == 'antpimaze':
        env_specific_params = dict(
            goal_threshold=0.5,
            max_trajectory_length=1000,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=0.5,
        )
    elif env_name == 'antomegamaze':
        env_specific_params = dict(
            goal_threshold=0.5,
            max_trajectory_length=1000,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=0.5,
        )
    elif env_name == '16rooms':
        env_specific_params = dict(
            goal_threshold=1,
            max_trajectory_length=1000,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=1,
        )
    elif env_name == 'ant9rooms':
        env_specific_params = dict(
            goal_threshold=1,
            max_trajectory_length=1000,
            max_timesteps=4e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=1,
        )
    elif env_name == 'ant9roomscloseddoors':
        env_specific_params = dict(
            goal_threshold=1,
            max_trajectory_length=1000,
            max_timesteps=4e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=1,
        )
    elif env_name == 'ant16rooms':
        env_specific_params = dict(
            goal_threshold=1,
            max_trajectory_length=1000,
            max_timesteps=1e6,
            eval_episodes=2,
            action_granularity=2,
            rrt_tree_node_cover_size=1,
        )
    elif env_name == 'ant16roomscloseddoors':
        env_specific_params = dict(
            goal_threshold=1,
            max_trajectory_length=1000,
            max_timesteps=1e6,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=1,
        )
    elif env_name == 'fetchslide':
        env_specific_params = dict(
            eval_freq=2000,
            goal_threshold=0.05,
            max_trajectory_length=100,
            max_timesteps=2e5,
            eval_episodes=2,
            action_granularity=5,
            rrt_tree_node_cover_size=0.1,
        )
    else:
        raise NotImplementedError()

    base_params.update(env_specific_params)
    return base_params
