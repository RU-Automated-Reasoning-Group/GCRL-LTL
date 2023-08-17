import os.path as osp

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Pusher2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    JOINT_INDS = list(range(0, 3))
    PUCK_INDS = list(range(3, 5))
    TARGET_INDS = list(range(5, 7))

    def __init__(self,
                 xml_file=os.path.dirname(os.path.realpath(__file__)) + '/../assets/pusher_2d.xml'
                 goal=(0, -1),
                 arm_object_distance_cost_coeff=0,
                 goal_object_distance_cost_coeff=1.0,
                 ctrl_cost_coeff=0.1):
        utils.EzPickle.__init__(**locals())

        self._goal_mask = [coordinate != 'any' for coordinate in goal]
        self._goal = np.array(goal)[self._goal_mask].astype(np.float32)

        self._arm_object_distance_cost_coeff = arm_object_distance_cost_coeff
        self._goal_object_distance_cost_coeff = goal_object_distance_cost_coeff
        self._ctrl_cost_coeff = ctrl_cost_coeff

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        self.model.stat.extent = 10

    def step(self, action):
        reward, info = self.compute_reward(self._get_obs(), action)

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        done = False

        return observation, reward, done, info

    def compute_reward(self, observations, actions):
        is_batch = True
        if observations.ndim == 1:
            observations = observations[None]
            actions = actions[None]
            is_batch = False

        arm_pos = observations[:, -6:-3]
        obj_pos = observations[:, -3:]
        obj_pos_masked = obj_pos[:, :2][:, self._goal_mask]

        #print(arm_pos)  # x y z=-0.075
        #print(obj_pos)  # x y z=-0.1
        #print(obj_pos_masked)  # x y
        #print('goal', self._goal[None])  # x=0 y=-1

        goal_object_distances = np.linalg.norm(
            self._goal[None] - obj_pos_masked, axis=1)

        # down -> y
        # left -> x
        goal_object_distances_x = (self._goal[None] - obj_pos_masked).squeeze()[0]        
        goal_object_distances_y = (self._goal[None] - obj_pos_masked).squeeze()[1]

        arm_object_distances = np.linalg.norm(arm_pos - obj_pos, axis=1)
        ctrl_costs = np.sum(actions**2, axis=1)

        costs = (
            + self._arm_object_distance_cost_coeff * arm_object_distances
            + self._goal_object_distance_cost_coeff * goal_object_distances
            + self._ctrl_cost_coeff * ctrl_costs)

        rewards = -costs

        if not is_batch:
            rewards = rewards.squeeze()
            arm_object_distances = arm_object_distances.squeeze()
            goal_object_distances = goal_object_distances.squeeze()

        return rewards, {
            'arm_object_distance': arm_object_distances,
            'goal_object_distance': goal_object_distances
        }

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        cam_pos = np.array([0, 0, 0, 4, -45, 0])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset_model(self, qpos=None, qvel=None):
        if qpos is None:
            qpos = np.random.uniform(
                low=-0.1, high=0.1, size=self.model.nq
            ) + self.init_qpos.squeeze()

            qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

            puck_position = np.random.uniform(
                low=[0.3, -1.0], high=[1.0, -0.4]),

            qpos[self.PUCK_INDS] = puck_position

        if qvel is None:
            qvel = self.init_qvel.copy().squeeze()
            qvel[self.PUCK_INDS] = 0
            qvel[self.TARGET_INDS] = 0

        # TODO: remnants from rllab -> gym conversion
        # qacc = np.zeros(self.sim.data.qacc.shape[0])
        # ctrl = np.zeros(self.sim.data.ctrl.shape[0])
        # full_state = np.concatenate((qpos, qvel, qacc, ctrl))

        self.set_state(np.array(qpos), np.array(qvel))

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            np.sin(self.sim.data.qpos.flat[self.JOINT_INDS]),
            np.cos(self.sim.data.qpos.flat[self.JOINT_INDS]),
            self.sim.data.qvel.flat[self.JOINT_INDS],
            self.get_body_com("distal_4"),
            self.get_body_com("object"),
        ]).reshape(-1)

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)


class Pusher2dLeftEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    JOINT_INDS = list(range(0, 3))
    PUCK_INDS = list(range(3, 5))
    TARGET_INDS = list(range(5, 7))

    def __init__(self,
                 xml_file=os.path.dirname(os.path.realpath(__file__)) + '/../assets/pusher_2d.xml',
                 goal=(0, -1),
                 ctrl_cost_coeff=0.1):
        utils.EzPickle.__init__(**locals())

        self._goal_mask = [coordinate != 'any' for coordinate in goal]
        self._goal = np.array(goal)[self._goal_mask].astype(np.float32)

        self._ctrl_cost_coeff = ctrl_cost_coeff

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        self.model.stat.extent = 10

    # under construction
    # dirty implementation
    def step(self, action):

        # before
        observation = self._get_obs()
        obj_pos = observation[-3:]
        obj_pos_masked = obj_pos[:2][self._goal_mask]
        x_position_before = obj_pos_masked[0]

        self.do_simulation(action, self.frame_skip)
        
        # after
        observation = self._get_obs()
        obj_pos = observation[-3:]
        obj_pos_masked = obj_pos[:2][self._goal_mask]
        x_position_after = obj_pos_masked[0]
        
        # x velocity
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        
        ctrl_costs = np.sum(action**2)
        
        reward = x_velocity * (-1.0) - self._ctrl_cost_coeff * ctrl_costs

        # under construction
        done = False
        info = {}

        return observation, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        cam_pos = np.array([0, 0, 0, 4, -45, 0])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset_model(self, qpos=None, qvel=None):
        if qpos is None:
            qpos = np.random.uniform(
                low=-0.1, high=0.1, size=self.model.nq
            ) + self.init_qpos.squeeze()

            qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

            puck_position = np.random.uniform(
                low=[0.3, -1.0], high=[1.0, -0.4]),

            qpos[self.PUCK_INDS] = puck_position

        if qvel is None:
            qvel = self.init_qvel.copy().squeeze()
            qvel[self.PUCK_INDS] = 0
            qvel[self.TARGET_INDS] = 0

        # TODO: remnants from rllab -> gym conversion
        # qacc = np.zeros(self.sim.data.qacc.shape[0])
        # ctrl = np.zeros(self.sim.data.ctrl.shape[0])
        # full_state = np.concatenate((qpos, qvel, qacc, ctrl))

        self.set_state(np.array(qpos), np.array(qvel))

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            np.sin(self.sim.data.qpos.flat[self.JOINT_INDS]),
            np.cos(self.sim.data.qpos.flat[self.JOINT_INDS]),
            self.sim.data.qvel.flat[self.JOINT_INDS],
            self.get_body_com("distal_4"),
            self.get_body_com("object"),
        ]).reshape(-1)

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)


class Pusher2dDownEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    JOINT_INDS = list(range(0, 3))
    PUCK_INDS = list(range(3, 5))
    TARGET_INDS = list(range(5, 7))

    def __init__(self,
                 xml_file=os.path.dirname(os.path.realpath(__file__)) + '/../assets/pusher_2d.xml',
                 goal=(0, -1),
                 ctrl_cost_coeff=0.1):
        utils.EzPickle.__init__(**locals())

        self._goal_mask = [coordinate != 'any' for coordinate in goal]
        self._goal = np.array(goal)[self._goal_mask].astype(np.float32)

        self._ctrl_cost_coeff = ctrl_cost_coeff

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        self.model.stat.extent = 10

    def step(self, action):

        # before
        observation = self._get_obs()
        obj_pos = observation[-3:]
        obj_pos_masked = obj_pos[:2][self._goal_mask]
        y_position_before = obj_pos_masked[1]

        self.do_simulation(action, self.frame_skip)
        
        # after
        observation = self._get_obs()
        obj_pos = observation[-3:]
        obj_pos_masked = obj_pos[:2][self._goal_mask]
        y_position_after = obj_pos_masked[1]
        
        # x velocity
        y_velocity = ((y_position_after - y_position_before)
                      / self.dt)
        
        ctrl_costs = np.sum(action**2)
        
        reward = y_velocity * (-1.0) - self._ctrl_cost_coeff * ctrl_costs

        # under construction
        done = False
        info = {}

        return observation, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        cam_pos = np.array([0, 0, 0, 4, -45, 0])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset_model(self, qpos=None, qvel=None):
        if qpos is None:
            qpos = np.random.uniform(
                low=-0.1, high=0.1, size=self.model.nq
            ) + self.init_qpos.squeeze()

            qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

            puck_position = np.random.uniform(
                low=[0.3, -1.0], high=[1.0, -0.4]),

            qpos[self.PUCK_INDS] = puck_position

        if qvel is None:
            qvel = self.init_qvel.copy().squeeze()
            qvel[self.PUCK_INDS] = 0
            qvel[self.TARGET_INDS] = 0

        # TODO: remnants from rllab -> gym conversion
        # qacc = np.zeros(self.sim.data.qacc.shape[0])
        # ctrl = np.zeros(self.sim.data.ctrl.shape[0])
        # full_state = np.concatenate((qpos, qvel, qacc, ctrl))

        self.set_state(np.array(qpos), np.array(qvel))

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            np.sin(self.sim.data.qpos.flat[self.JOINT_INDS]),
            np.cos(self.sim.data.qpos.flat[self.JOINT_INDS]),
            self.sim.data.qvel.flat[self.JOINT_INDS],
            self.get_body_com("distal_4"),
            self.get_body_com("object"),
        ]).reshape(-1)

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)
