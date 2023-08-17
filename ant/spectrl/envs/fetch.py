import os
import mujoco_py

from gym import utils
from gym.envs.robotics import fetch_env
from copy import deepcopy


# Ensure we get the path separator correct on windows
PICK_MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')
REACH_MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')
PUSH_MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchEnvWithStates(fetch_env.FetchEnv):

    def get_sim_state(self):
        return deepcopy(self.sim.get_state())

    def set_sim_state(self, state):
        self.sim.reset()
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, state.qpos, state.qvel,
                                         state.act, state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
        return self._get_obs()


class FetchPickAndPlaceEnv(FetchEnvWithStates, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, PICK_MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


class FetchReachEnv(FetchEnvWithStates, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, REACH_MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


class FetchPushEnv(FetchEnvWithStates, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, PUSH_MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
