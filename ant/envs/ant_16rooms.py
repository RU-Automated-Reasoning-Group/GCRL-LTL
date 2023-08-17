from argparse import _get_action_name
from envs.goal_env import GoalEnv
from collections import OrderedDict
from envs.ant.ant_maze_env import AntMazeEnv
import numpy as np
import gym
from gym import spaces
from multiworld.core.serializable import Serializable
import matplotlib.pyplot as plt
import random
import dill as pickle
import getopt
import os
import sys
from dependencies.multiworld.core.serializable import Serializable
from PIL import Image
import torch
import torch.nn as nn
from algo.utils import Line, Box_obstacle
import math
import matplotlib.patches as patches
from algo.utils import *


class Ant16RoomsEnv(GoalEnv, Serializable):
    def __init__(self, silent=True):
        cls = AntMazeEnv

        maze_id = 'Ant16Rooms'
        n_bins = 0
        observe_blocks = False
        put_spin_near_agent = False
        top_down_view = False
        manual_collision = False
        maze_size_scaling = 1

        gym_mujoco_kwargs = {
            'maze_id': maze_id,
            'maze_height': 4,
            'n_bins': n_bins,
            'observe_blocks': observe_blocks,
            'put_spin_near_agent': put_spin_near_agent,
            'top_down_view': top_down_view,
            'manual_collision': manual_collision,
            'maze_size_scaling': maze_size_scaling
        }

        self.maze_env = cls(**gym_mujoco_kwargs)  # wrapped_env
        self.maze_env.reset()

        # to scale room position from original 16 rooms env to ant16rooms env
        self.room_scaling = 9

        self.maze_structure = self.maze_env.MAZE_STRUCTURE
        # structure = [
        #     [1, 1,1,1,1,  1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,'r',0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1] (x)
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 1,1,0,0,  0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1]

        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,1,  1,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,1,  1,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 1,1,0,0,  0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1]

        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,1,1,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,1,1,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 1,1,0,0,  0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1]

        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 0,0,0,0,  0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1]
        #     [1, 1,1,1,1,  1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1]
        #              (y)
        # ]
        self.quick_init(locals())

        # all primitive policies
        # up = along positive x-axis; down = along negative x-axis; left = along positive y-axis; right = along negative y-axis
        self.primitives = []
        for direction in ['up', 'down', 'left', 'right']:
            gcsl_dir = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.realpath(__file__))))
            filename = gcsl_dir + '/gcsl_merge/primitives/ant_low_torque/' + direction + '.pt'
            # filename = '/root/code/gcsl_ant/primitives/ant_low_torque/' + direction + '.pt'
            self.primitives.append(torch.load(filename))
            # make it corresponding to the actual xy-axis
        self.low_torque_limit = torch.ones(
            self.lowlevel_action_space.shape[0])*30

        # obstacles region:
        self.ant_radius = 0.75
        self.wall_collision_buffer = 0.25
        self.obstacles = []

        obstacle_0_1 = Box_obstacle([-0.5, 10.5], [1.5, 10.5], [-0.5, 8.5], [
                                    1.5, 8.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(obstacle_0_1)

        obstacle_3_2 = Box_obstacle([26.5, 19.5], [28.5, 19.5], [26.5, 17.5], [
                                    28.5, 17.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(obstacle_3_2)

        cross_vertical_11 = Box_obstacle([4.5, 7.5], [5.5, 7.5], [4.5, 2.5], [
                                         5.5, 2.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_11 = Box_obstacle([2.5, 5.5], [7.5, 5.5], [2.5, 4.5], [
                                           7.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_11)
        self.obstacles.append(cross_horizontal_11)

        cross_vertical_12 = Box_obstacle([4.5, 16.5], [5.5, 16.5], [4.5, 11.5], [
                                         5.5, 11.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_12 = Box_obstacle([2.5, 14.5], [7.5, 14.5], [2.5, 13.5], [
                                           7.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_12)
        self.obstacles.append(cross_horizontal_12)

        cross_vertical_13 = Box_obstacle([4.5, 25.5], [5.5, 25.5], [4.5, 20.5], [
                                         5.5, 20.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_13 = Box_obstacle([2.5, 23.5], [7.5, 23.5], [2.5, 22.5], [
                                           7.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_13)
        self.obstacles.append(cross_horizontal_13)

        cross_vertical_21 = Box_obstacle([13.5, 7.5], [14.5, 7.5], [13.5, 2.5], [
                                         14.5, 2.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_21 = Box_obstacle([11.5, 5.5], [16.5, 5.5], [11.5, 4.5], [
                                           16.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_21)
        self.obstacles.append(cross_horizontal_21)

        cross_vertical_22 = Box_obstacle([13.5, 16.5], [14.5, 16.5], [13.5, 11.5], [
                                         14.5, 11.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_22 = Box_obstacle([11.5, 14.5], [16.5, 14.5], [11.5, 13.5], [
                                           16.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_22)
        self.obstacles.append(cross_horizontal_22)

        cross_vertical_23 = Box_obstacle([13.5, 25.5], [14.5, 25.5], [13.5, 20.5], [
                                         14.5, 20.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_23 = Box_obstacle([11.5, 23.5], [16.5, 23.5], [11.5, 22.5], [
                                           16.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_23)
        self.obstacles.append(cross_horizontal_23)

        cross_vertical_31 = Box_obstacle([22.5, 7.5], [23.5, 7.5], [22.5, 2.5], [
                                         23.5, 2.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_31 = Box_obstacle([20.5, 5.5], [25.5, 5.5], [20.5, 4.5], [
                                           25.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_31)
        self.obstacles.append(cross_horizontal_31)

        cross_vertical_32 = Box_obstacle([22.5, 16.5], [23.5, 16.5], [22.5, 11.5], [
                                         23.5, 11.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_32 = Box_obstacle([20.5, 14.5], [25.5, 14.5], [20.5, 13.5], [
                                           25.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_32)
        self.obstacles.append(cross_horizontal_32)

        cross_vertical_33 = Box_obstacle([22.5, 25.5], [23.5, 25.5], [22.5, 20.5], [
                                         23.5, 20.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_33 = Box_obstacle([20.5, 23.5], [25.5, 23.5], [20.5, 22.5], [
                                           25.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_33)
        self.obstacles.append(cross_horizontal_33)

        left_1 = Box_obstacle([-3.5, 5.5], [-1.5, 5.5], [-3.5, 4.5],
                              [-1.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        left_2 = Box_obstacle([-3.5, 14.5], [-1.5, 14.5], [-3.5, 13.5],
                              [-1.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        left_3 = Box_obstacle([-3.5, 23.5], [-1.5, 23.5], [-3.5, 22.5],
                              [-1.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(left_1)
        self.obstacles.append(left_2)
        self.obstacles.append(left_3)

        right_1 = Box_obstacle([29.5, 5.5], [31.5, 5.5], [29.5, 4.5], [
                               31.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        right_2 = Box_obstacle([29.5, 14.5], [31.5, 14.5], [29.5, 13.5], [
                               31.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        right_3 = Box_obstacle([29.5, 23.5], [31.5, 23.5], [29.5, 22.5], [
                               31.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(right_1)
        self.obstacles.append(right_2)
        self.obstacles.append(right_3)

        top_1 = Box_obstacle([4.5, -1.5], [5.5, -1.5], [4.5, -3.5],
                             [5.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        top_2 = Box_obstacle([13.5, -1.5], [14.5, -1.5], [13.5, -3.5],
                             [14.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        top_3 = Box_obstacle([22.5, -1.5], [23.5, -1.5], [22.5, -3.5],
                             [23.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(top_1)
        self.obstacles.append(top_2)
        self.obstacles.append(top_3)

        bottom_1 = Box_obstacle([4.5, 29.5], [5.5, 29.5], [4.5, 27.5], [
                                5.5, 27.5], self.ant_radius + self.wall_collision_buffer)
        bottom_2 = Box_obstacle([13.5, 29.5], [14.5, 29.5], [13.5, 27.5], [
                                14.5, 27.5], self.ant_radius + self.wall_collision_buffer)
        bottom_3 = Box_obstacle([22.5, 29.5], [23.5, 29.5], [22.5, 27.5], [
                                23.5, 27.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(bottom_1)
        self.obstacles.append(bottom_2)
        self.obstacles.append(bottom_3)

        left_border = Box_obstacle([-4.5, 32.5], [-3.5, 32.5], [-4.5, -4.5],
                                   [-3.5, -4.5], self.ant_radius + self.wall_collision_buffer)
        right_border = Box_obstacle([32.5, 32.5], [32.5, 32.5], [
                                    31.5, -4.5], [31.5, -4.5], self.ant_radius + self.wall_collision_buffer)
        upper_border = Box_obstacle([-4.5, -3.5], [32.5, -3.5], [-4.5, -4.5], [
                                    32.5, -4.5], self.ant_radius + self.wall_collision_buffer)
        bottom_border = Box_obstacle([-4.5, 32.5], [32.5, 32.5], [-4.5, 31.5], [
                                     32.5, 31.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(left_border)
        self.obstacles.append(right_border)
        self.obstacles.append(upper_border)
        self.obstacles.append(bottom_border)

    @staticmethod
    def get_action_from_primitive(model, x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x, deterministic=True)
        return torch.from_numpy(action)

    @property
    def observation_space(self):
        shape = self.maze_env._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    # action space is a 4-dim vector which indicate four primitives
    def action_space(self):
        low = np.zeros((4))
        high = np.ones((4))
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def lowlevel_action_space(self):
        return self.maze_env.action_space

    @property
    def state_space(self):
        return self.observation_space

    @property
    def goal_space(self):
        return spaces.Box(low=np.array([-3, -3]), high=np.array([31, 31]), dtype=np.float32)

    # actaully, the data it returns is tof the same shape of state, but just contain the final goal information
    def sample_goal(self):
        shape = self.observation_space.shape
        goal = np.zeros(shape)

        # do sample until goal is a feasible one
        goal_candidate = self.goal_space.sample()
        goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                         int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        while (self.maze_structure[goal_maze_idx[0] + 4][goal_maze_idx[1] + 4] == 1
               or self.goal_in_obstacle([goal_candidate[0], goal_candidate[1]]) == True):
            goal_candidate = self.goal_space.sample()
            goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                             int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        goal[: 2] = goal_candidate
        return goal

    def goal_in_obstacle(self, goal):
        for box_obst in self.obstacles:
            if box_obst.within_box(goal):
                return True
        return False

    def intersect_with_obstacle(self, line):
        return False

    def reset(self):
        """
        Resets the environment and returns a state vector
        Returns: The initial state
        """
        ant_obs = self.maze_env.reset()
        self.current_state = ant_obs
        self.previous_state = self.current_state
        self.weights = []
        self.actions = []
        return ant_obs

    def reset_with_init_pos_at(self, robot_maze_pos):
        ant_obs = self.maze_env.reset_with_init_pos_at(robot_maze_pos)
        self.current_state = ant_obs
        self.previous_state = self.current_state
        self.weights = []
        self.actions = []
        return ant_obs

    # param "h igh_action" representing distribution of primitives
    # (highlevel_action from policy strictly means: up is go up, down is go down, etc)
    def step(self, highlevel_action):
        """
        Runs 1 step of simulation

        Returns:
            A tuple containing:
                next_state
                reward (always 0)
                done
                infos
        """
        # print("highlevel_action:"+str(highlevel_action))
        normalized_highlevel_action = np.array(
            [highlevel_action[3], highlevel_action[2], highlevel_action[1], highlevel_action[0]], dtype=np.float32)
        sum = np.sum(normalized_highlevel_action)
        if sum > 0:
            normalized_highlevel_action = normalized_highlevel_action / \
                float(sum)
        self.weights.append(normalized_highlevel_action)

        action = np.zeros(self.lowlevel_action_space.shape, dtype=np.float32)
        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            # f.write("primitive action:")
            for i in range(len(normalized_highlevel_action)):
                primitive_input = self.current_state[range(2, 29)]
                action_ = normalized_highlevel_action[i] * self.get_action_from_primitive(
                    self.primitives[i], primitive_input[np.newaxis, :]).numpy().ravel()
                # f.write(str(self.get_action_from_primitive(self.primitives[i], primitive_input).numpy()))
                action += action_
        # action = np.clip(action, self.low_torque_limit*-1, self.low_torque_limit)
        self.actions.append(action)

        ob, reward, done, info = self.maze_env.step(action)

        self.previous_state = self.current_state
        self.current_state = ob
        done = False

        # if (self.goal_in_obstacle(self.previous_state[:2]) == False) and (
        #         self.goal_in_obstacle(self.current_state[:2]) == False) and (
        #         self.intersect_with_obstacle(Line(self.previous_state[:2], self.current_state[:2])) == False):

        # if (self.goal_in_obstacle(self.previous_state[:2]) == False) and (self.goal_in_obstacle(self.current_state[:2]) == False):
        #     info['expandable'] = True
        # else:
        #     info['expandable'] = False
        if self.goal_in_obstacle(self.current_state[:2]) == True:
            reward = -1
        else:
            reward = 0

        info['expandable'] = True
        # distance = np.linalg.norm(ob[:2] - np.array([0, 16]))
        # reward = -distance
        # done = True if distance < 1.0 else False
        # progress = (1 - (distance - 1) / (16 - 1)) * 100

        # info['distance'] = (distance - 1) / (16 - 1) if distance > 1.0 else 0.0
        # info['progress'] = progress

        return ob, reward, done, info

    ###
    def observation(self, state):
        """
        Returns the observation for a given state

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        return state

    # extract goal info from given state (actually the state/position ant has reached)
    def extract_goal(self, state):
        """
        Returns the goal representation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        return state[..., : 2]

    ###
    def _extract_sgoal(self, state):
        return state[..., : 2]

    def goal_distance(self, state, goal_state):
        # self.goal_metric = 'euclidean':
        diff = self.extract_goal(state)-self.extract_goal(goal_state)
        return np.linalg.norm(diff, axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states, success_vec=[], prefix="", all_reschedule_points=[]):
        """
        Gets things to log
        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]
        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """
        # with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
        #     for i in range(len(trajectories)):
        #         f.write("Eval try AntCross Maze %d: Path from ant(0,0) to goal(0,16)\n:" % (i))
        #         for j in range(len(trajectories[i])):
        #             trajectory = "{"+str(self.weights[j])+str(self.actions[j])+str(trajectories[i][j][:3])+"}\n"
        #             f.write(trajectory)
        # f.close()

        for i in range(len(trajectories)):
            x = []
            y = []
            reschedule_points_x = []
            reschedule_points_y = []
            for j in range(len(trajectories[i])):
                x.append(trajectories[i][j][0] + 4 *
                         self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 4 *
                         self.maze_env.MAZE_SIZE_SCALING)
            if len(all_reschedule_points[i]) > 0:
                for k in range(len(all_reschedule_points[i])):
                    reschedule_points_x.append(all_reschedule_points[i][k][0] + 4 *
                                               self.maze_env.MAZE_SIZE_SCALING)
                    reschedule_points_y.append(all_reschedule_points[i][k][1] + 4 *
                                               self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(
                    x, y, desired_goal_states[i][0: 2], True, reschedule_points_x=reschedule_points_x, reschedule_points_y=reschedule_points_y)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace" +
                            prefix+str(i)+".png", dpi=500)
            else:
                self.plot_trajectory_fig(
                    x, y, desired_goal_states[i][0: 2], False, reschedule_points_x=reschedule_points_x, reschedule_points_y=reschedule_points_y)
                plt.savefig("./code/gcsl_ant/fig/trace" +
                            prefix + str(i)+".png", dpi=500)

        return OrderedDict([])

    def plot_trajectory_fig(self, x, y, desired_goal, reached=False, reschedule_points_x=[], reschedule_points_y=[]):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        # Draw goal
        ax.scatter(desired_goal[0] + 4 * maze_scaling,
                   desired_goal[1] + 4 * maze_scaling, s=400, marker='*')

        # Draw starting position
        ax.scatter(x[0], y[0], s=50, marker='D', color='k')

        # Draw path
        if reached == True:
            ax.plot(x, y, color="r")
        else:
            ax.plot(x, y, color="k")  # black

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        for i in range(len(reschedule_points_x)):
            ax.scatter(
                reschedule_points_x[i], reschedule_points_y[i], s=100, marker='d', color='orange')

    def sample_goal_scatter_fig(self, x, y):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        assert len(x) == len(y)

        # Draw goal
        for i in range(len(x)):
            ax.scatter(x[i] + 4 * maze_scaling, y[i] +
                       4 * maze_scaling, s=10, marker='*')

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure,
                            self.maze_env.MAZE_SIZE_SCALING)

        plt.savefig("./code/gcsl_ant/fig/" + self.maze_env._maze_id +
                    "sample_goal_scatter_"+str(self.maze_env._maze_id)+".pdf")

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, rrt_tree, opt_path=[], states=[], count=-1):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        px = [x + 4 * maze_scaling for x, y in rrt_tree.vertices]
        py = [y + 4 * maze_scaling for x, y in rrt_tree.vertices]
        fig, ax = plt.subplots()
        ax.scatter(px, py, c='cyan')
        ax.scatter([x + 4 * maze_scaling for x in goal_pos_x[-goal_scatter_num:]],
                   [y + 4 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*', c='r', zorder=1)

        lines = []
        for edge in rrt_tree.edges:
            node1 = rrt_tree.vertices[edge[0]]
            node2 = rrt_tree.vertices[edge[1]]
            node1 = (node1[0] + 4 * maze_scaling, node1[1] + 4 * maze_scaling)
            node2 = (node2[0] + 4 * maze_scaling, node2[1] + 4 * maze_scaling)
            lines.append((node1, node2))
        lc = mc.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)

        if len(opt_path) > 0:

            path = []
            for i in range(len(opt_path)-1):
                node1 = rrt_tree.vertices[opt_path[i]]
                node2 = rrt_tree.vertices[opt_path[i + 1]]
                node1 = (node1[0] + 4 * maze_scaling,
                         node1[1] + 4 * maze_scaling)
                node2 = (node2[0] + 4 * maze_scaling,
                         node2[1] + 4 * maze_scaling)
                path.append((node1, node2))
            lc2 = mc.LineCollection(path, colors='red', linewidths=1)
            ax.add_collection(lc2)

        if len(states) > 0:
            trace_x = []
            trace_y = []
            for i in range(len(states)):
                trace_x.append(states[i][0] + 4 * maze_scaling)
                trace_y.append(states[i][1] + 4 * maze_scaling)
            ax.plot(trace_x, trace_y, color="k", linewidth=1)

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        plt.savefig("./code/gcsl_ant/fig/rrt_star_tree" +
                    str(timestep) + "_"+str(count) + ".pdf")

    def save_waypoints(self, waypoints, prefix):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        x = []
        y = []
        # Draw goal
        for waypoint in waypoints:
            ax.scatter(waypoint[0] + 4 * maze_scaling,
                       waypoint[1] + 4 * maze_scaling, s=400, marker='*')
            x.append(waypoint[0]+4 * maze_scaling)
            y.append(waypoint[1]+4 * maze_scaling)

        # Draw path
        ax.plot(x, y, color="k")

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)
        plt.savefig("./code/gcsl_ant/fig/" + str(prefix) +
                    "way_points_"+str(self.maze_env._maze_id)+".pdf")

    # for maze info checking [mws]

    def print_maze_infos(self):
        ant_obs = self.maze_env._get_obs()
        # ant_obs and shape
        print(ant_obs.shape)
        print(dir(self.maze_env))
        print(self.maze_env.observation_space)
        print(self.maze_env.action_space)
        # print(self.maze_env.MAZE_STRUCTURE)
        # print(self.maze_env.MAZE_HEIGHT)
        print(self.maze_env._find_robot())
        print(self.maze_env._init_positions)


class Ant16RoomswithClosedDoorsEnv(GoalEnv, Serializable):
    def __init__(self, silent=True):
        cls = AntMazeEnv

        maze_id = 'Ant16RoomsClosedDoors'
        n_bins = 0
        observe_blocks = False
        put_spin_near_agent = False
        top_down_view = False
        manual_collision = False
        maze_size_scaling = 1

        gym_mujoco_kwargs = {
            'maze_id': maze_id,
            'maze_height': 4,
            'n_bins': n_bins,
            'observe_blocks': observe_blocks,
            'put_spin_near_agent': put_spin_near_agent,
            'top_down_view': top_down_view,
            'manual_collision': manual_collision,
            'maze_size_scaling': maze_size_scaling
        }

        self.maze_env = cls(**gym_mujoco_kwargs)  # wrapped_env
        self.maze_env.reset()

        # to scale room position from original 16 rooms env to ant16rooms env
        self.room_scaling = 9

        self.maze_structure = self.maze_env.MAZE_STRUCTURE
        # structure = [
        #     [1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,'r',0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],(x)
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 1,1,0,0,0,0,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,0,0,0,0,1,1, 1],

        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,1,1,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,1,1,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 1,1,0,0,0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1, 1,1,1,1,1,1,1,1, 1],

        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,1,1,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,1,1,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 1,1,0,0,0,0,1,1, 1, 1,1,0,0,0,0,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,0,0,0,0,1,1, 1],

        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1, 0,0,0,0,0,0,0,0, 1],
        #     [1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1, 1,1,1,1,1,1,1,1, 1],
        #              (y)
        # ]
        self.quick_init(locals())

        # all primitive policies
        # up = along positive x-axis; down = along negative x-axis; left = along positive y-axis; right = along negative y-axis
        self.primitives = []
        for direction in ['up', 'down', 'left', 'right']:
            gcsl_dir = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.realpath(__file__))))
            filename = gcsl_dir + '/primitives/ant_low_torque/' + direction + '.pt'
            # filename = '/root/code/gcsl_ant/primitives/ant_low_torque/' + direction + '.pt'
            self.primitives.append(torch.load(filename))
            # make it corresponding to the actual xy-axis
        self.low_torque_limit = torch.ones(
            self.lowlevel_action_space.shape[0])*30

        # obstacles region:
        self.ant_radius = 0.75
        self.wall_collision_buffer = 0.25
        self.obstacles = []

        obstacle_0_1 = Box_obstacle([-0.5, 10.5], [1.5, 10.5], [-0.5, 8.5], [
                                    1.5, 8.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(obstacle_0_1)

        obstacle_3_2 = Box_obstacle([26.5, 19.5], [28.5, 19.5], [26.5, 17.5], [
                                    28.5, 17.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(obstacle_3_2)

        cross_vertical_11 = Box_obstacle([4.5, 7.5], [5.5, 7.5], [4.5, 2.5], [
                                         5.5, 2.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_11 = Box_obstacle([2.5, 5.5], [11.5, 5.5], [2.5, 4.5], [
                                           11.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_11)
        self.obstacles.append(cross_horizontal_11)

        cross_vertical_12 = Box_obstacle([4.5, 16.5], [5.5, 16.5], [4.5, 11.5], [
                                         5.5, 11.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_12 = Box_obstacle([2.5, 14.5], [7.5, 14.5], [2.5, 13.5], [
                                           7.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_12)
        self.obstacles.append(cross_horizontal_12)

        cross_vertical_13 = Box_obstacle([4.5, 25.5], [5.5, 25.5], [4.5, 20.5], [
                                         5.5, 20.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_13 = Box_obstacle([2.5, 23.5], [7.5, 23.5], [2.5, 22.5], [
                                           7.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_13)
        self.obstacles.append(cross_horizontal_13)

        cross_vertical_21 = Box_obstacle([13.5, 7.5], [14.5, 7.5], [13.5, 2.5], [
                                         14.5, 2.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_21 = Box_obstacle([11.5, 5.5], [20.5, 5.5], [11.5, 4.5], [
                                           20.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_21)
        self.obstacles.append(cross_horizontal_21)

        cross_vertical_22 = Box_obstacle([13.5, 16.5], [14.5, 16.5], [13.5, 11.5], [
                                         14.5, 11.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_22 = Box_obstacle([11.5, 14.5], [16.5, 14.5], [11.5, 13.5], [
                                           16.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_22)
        self.obstacles.append(cross_horizontal_22)

        cross_vertical_23 = Box_obstacle([13.5, 25.5], [14.5, 25.5], [13.5, 20.5], [
                                         14.5, 20.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_23 = Box_obstacle([11.5, 23.5], [20.5, 23.5], [11.5, 22.5], [
                                           20.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_23)
        self.obstacles.append(cross_horizontal_23)

        cross_vertical_31 = Box_obstacle([22.5, 7.5], [23.5, 7.5], [22.5, 2.5], [
                                         23.5, 2.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_31 = Box_obstacle([20.5, 5.5], [29.5, 5.5], [20.5, 4.5], [
                                           29.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_31)
        self.obstacles.append(cross_horizontal_31)

        cross_vertical_32 = Box_obstacle([22.5, 16.5], [23.5, 16.5], [22.5, 11.5], [
                                         23.5, 11.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_32 = Box_obstacle([20.5, 14.5], [25.5, 14.5], [20.5, 13.5], [
                                           25.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_32)
        self.obstacles.append(cross_horizontal_32)

        cross_vertical_33 = Box_obstacle([22.5, 25.5], [23.5, 25.5], [22.5, 20.5], [
                                         23.5, 20.5], self.ant_radius + self.wall_collision_buffer)
        cross_horizontal_33 = Box_obstacle([20.5, 23.5], [25.5, 23.5], [20.5, 22.5], [
                                           25.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(cross_vertical_33)
        self.obstacles.append(cross_horizontal_33)

        left_1 = Box_obstacle([-3.5, 5.5], [-1.5, 5.5], [-3.5, 4.5],
                              [-1.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        left_2 = Box_obstacle([-3.5, 14.5], [-1.5, 14.5], [-3.5, 13.5],
                              [-1.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        left_3 = Box_obstacle([-3.5, 23.5], [-1.5, 23.5], [-3.5, 22.5],
                              [-1.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(left_1)
        self.obstacles.append(left_2)
        self.obstacles.append(left_3)

        right_1 = Box_obstacle([29.5, 5.5], [31.5, 5.5], [29.5, 4.5], [
                               31.5, 4.5], self.ant_radius + self.wall_collision_buffer)
        right_2 = Box_obstacle([29.5, 14.5], [31.5, 14.5], [29.5, 13.5], [
                               31.5, 13.5], self.ant_radius + self.wall_collision_buffer)
        right_3 = Box_obstacle([29.5, 23.5], [31.5, 23.5], [29.5, 22.5], [
                               31.5, 22.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(right_1)
        self.obstacles.append(right_2)
        self.obstacles.append(right_3)

        top_1 = Box_obstacle([4.5, -1.5], [5.5, -1.5], [4.5, -3.5],
                             [5.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        top_2 = Box_obstacle([13.5, -1.5], [14.5, -1.5], [13.5, -3.5],
                             [14.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        top_3 = Box_obstacle([22.5, -1.5], [23.5, -1.5], [22.5, -3.5],
                             [23.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(top_1)
        self.obstacles.append(top_2)
        self.obstacles.append(top_3)

        bottom_1 = Box_obstacle([4.5, 29.5], [5.5, 29.5], [4.5, 27.5], [
                                5.5, 27.5], self.ant_radius + self.wall_collision_buffer)
        bottom_2 = Box_obstacle([13.5, 29.5], [14.5, 29.5], [13.5, 27.5], [
                                14.5, 27.5], self.ant_radius + self.wall_collision_buffer)
        bottom_3 = Box_obstacle([22.5, 29.5], [23.5, 29.5], [22.5, 27.5], [
                                23.5, 27.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(bottom_1)
        self.obstacles.append(bottom_2)
        self.obstacles.append(bottom_3)

        left_border = Box_obstacle([-4.5, 32.5], [-3.5, 32.5], [-4.5, -4.5],
                                   [-3.5, -4.5], self.ant_radius + self.wall_collision_buffer)
        right_border = Box_obstacle([32.5, 32.5], [32.5, 32.5], [
                                    31.5, -4.5], [31.5, -4.5], self.ant_radius + self.wall_collision_buffer)
        upper_border = Box_obstacle([-4.5, -3.5], [32.5, -3.5], [-4.5, -4.5], [
                                    32.5, -4.5], self.ant_radius + self.wall_collision_buffer)
        bottom_border = Box_obstacle([-4.5, 32.5], [32.5, 32.5], [-4.5, 31.5], [
                                     32.5, 31.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(left_border)
        self.obstacles.append(right_border)
        self.obstacles.append(upper_border)
        self.obstacles.append(bottom_border)

    @staticmethod
    def get_action_from_primitive(model, x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x, deterministic=True)
        return torch.from_numpy(action)

    @property
    def observation_space(self):
        shape = self.maze_env._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    # action space is a 4-dim vector which indicate four primitives
    def action_space(self):
        low = np.zeros((4))
        high = np.ones((4))
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def lowlevel_action_space(self):
        return self.maze_env.action_space

    @property
    def state_space(self):
        return self.observation_space

    @property
    def goal_space(self):
        return spaces.Box(low=np.array([-3, -3]), high=np.array([31, 31]), dtype=np.float32)

    # actaully, the data it returns is tof the same shape of state, but just contain the final goal information
    def sample_goal(self):
        shape = self.observation_space.shape
        goal = np.zeros(shape)

        # do sample until goal is a feasible one
        goal_candidate = self.goal_space.sample()
        goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                         int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        while (self.maze_structure[goal_maze_idx[0] + 4][goal_maze_idx[1] + 4] == 1
               or self.goal_in_obstacle([goal_candidate[0], goal_candidate[1]]) == True):
            goal_candidate = self.goal_space.sample()
            goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                             int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        goal[: 2] = goal_candidate
        return goal

    def goal_in_obstacle(self, goal):
        for box_obst in self.obstacles:
            if box_obst.within_box(goal):
                return True
        return False

    def intersect_with_obstacle(self, line):
        return False

    def reset(self):
        """
        Resets the environment and returns a state vector
        Returns: The initial state
        """
        ant_obs = self.maze_env.reset()
        self.current_state = ant_obs
        self.previous_state = self.current_state
        self.weights = []
        self.actions = []
        return ant_obs

    def reset_with_init_pos_at(self, robot_maze_pos):
        ant_obs = self.maze_env.reset_with_init_pos_at(robot_maze_pos)
        self.current_state = ant_obs
        self.previous_state = self.current_state
        self.weights = []
        self.actions = []
        return ant_obs

    # param "h igh_action" representing distribution of primitives
    # (highlevel_action from policy strictly means: up is go up, down is go down, etc)
    def step(self, highlevel_action):
        """
        Runs 1 step of simulation

        Returns:
            A tuple containing:
                next_state
                reward (always 0)
                done
                infos
        """
        # print("highlevel_action:"+str(highlevel_action))
        normalized_highlevel_action = np.array(
            [highlevel_action[3], highlevel_action[2], highlevel_action[1], highlevel_action[0]], dtype=np.float32)
        sum = np.sum(normalized_highlevel_action)
        if sum > 0:
            normalized_highlevel_action = normalized_highlevel_action / \
                float(sum)
        self.weights.append(normalized_highlevel_action)

        action = np.zeros(self.lowlevel_action_space.shape, dtype=np.float32)
        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            # f.write("primitive action:")
            for i in range(len(normalized_highlevel_action)):
                primitive_input = self.current_state[range(2, 29)]
                action_ = normalized_highlevel_action[i] * self.get_action_from_primitive(
                    self.primitives[i], primitive_input[np.newaxis, :]).numpy().ravel()
                # f.write(str(self.get_action_from_primitive(self.primitives[i], primitive_input).numpy()))
                action += action_
        # action = np.clip(action, self.low_torque_limit*-1, self.low_torque_limit)
        self.actions.append(action)

        ob, reward, done, info = self.maze_env.step(action)

        self.previous_state = self.current_state
        self.current_state = ob
        done = False

        # if (self.goal_in_obstacle(self.previous_state[:2]) == False) and (
        #         self.goal_in_obstacle(self.current_state[:2]) == False) and (
        #         self.intersect_with_obstacle(Line(self.previous_state[:2], self.current_state[:2])) == False):

        # if (self.goal_in_obstacle(self.previous_state[:2]) == False) and (self.goal_in_obstacle(self.current_state[:2]) == False):
        #     info['expandable'] = True
        # else:
        #     info['expandable'] = False
        if self.goal_in_obstacle(self.current_state[:2]) == True:
            reward = -1
        else:
            reward = 0

        info['expandable'] = True
        # distance = np.linalg.norm(ob[:2] - np.array([0, 16]))
        # reward = -distance
        # done = True if distance < 1.0 else False
        # progress = (1 - (distance - 1) / (16 - 1)) * 100

        # info['distance'] = (distance - 1) / (16 - 1) if distance > 1.0 else 0.0
        # info['progress'] = progress

        return ob, reward, done, info

    ###
    def observation(self, state):
        """
        Returns the observation for a given state

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        return state

    # extract goal info from given state (actually the state/position ant has reached)
    def extract_goal(self, state):
        """
        Returns the goal representation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        return state[..., : 2]

    ###
    def _extract_sgoal(self, state):
        return state[..., : 2]

    def goal_distance(self, state, goal_state):
        # self.goal_metric = 'euclidean':
        diff = self.extract_goal(state)-self.extract_goal(goal_state)
        return np.linalg.norm(diff, axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states, success_vec=[], prefix=""):
        """
        Gets things to log
        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]
        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """
        # with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
        #     for i in range(len(trajectories)):
        #         f.write("Eval try AntCross Maze %d: Path from ant(0,0) to goal(0,16)\n:" % (i))
        #         for j in range(len(trajectories[i])):
        #             trajectory = "{"+str(self.weights[j])+str(self.actions[j])+str(trajectories[i][j][:3])+"}\n"
        #             f.write(trajectory)
        # f.close()

        for i in range(len(trajectories)):
            x = []
            y = []
            for j in range(len(trajectories[i])):
                x.append(trajectories[i][j][0] + 4 *
                         self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 4 *
                         self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(
                    x, y, desired_goal_states[i][0: 2], True)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace" +
                            prefix+str(i)+".pdf")
            else:
                self.plot_trajectory_fig(
                    x, y, desired_goal_states[i][0: 2], False)
                plt.savefig("./code/gcsl_ant/fig/trace"+prefix + str(i)+".pdf")

        return OrderedDict([])

    def plot_trajectory_fig(self, ax, x, y, desired_goal, reached=False):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING

        plt.clf()
        fig, ax = plt.subplots()

        # Draw goal
        ax.scatter(desired_goal[0] + 4 * maze_scaling,
                   desired_goal[1] + 4 * maze_scaling, s=400, marker='*')

        # Draw path
        if reached == True:
            ax.plot(x, y, color="r")
        else:
            ax.plot(x, y, color="k")  # black

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

    def sample_goal_scatter_fig(self, x, y):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        assert len(x) == len(y)

        # Draw goal
        for i in range(len(x)):
            ax.scatter(x[i] + 4 * maze_scaling, y[i] +
                       4 * maze_scaling, s=10, marker='*')

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure,
                            self.maze_env.MAZE_SIZE_SCALING)

        plt.savefig("./code/gcsl_ant/fig/" + self.maze_env._maze_id +
                    "sample_goal_scatter_"+str(self.maze_env._maze_id)+".pdf")

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, rrt_tree, opt_path=[], states=[], count=-1):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        px = [x + 4 * maze_scaling for x, y in rrt_tree.vertices]
        py = [y + 4 * maze_scaling for x, y in rrt_tree.vertices]
        fig, ax = plt.subplots()
        ax.scatter(px, py, c='cyan')
        ax.scatter([x + 4 * maze_scaling for x in goal_pos_x[-goal_scatter_num:]],
                   [y + 4 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*', c='r', zorder=1)

        lines = []
        for edge in rrt_tree.edges:
            node1 = rrt_tree.vertices[edge[0]]
            node2 = rrt_tree.vertices[edge[1]]
            node1 = (node1[0] + 4 * maze_scaling, node1[1] + 4 * maze_scaling)
            node2 = (node2[0] + 4 * maze_scaling, node2[1] + 4 * maze_scaling)
            lines.append((node1, node2))
        lc = mc.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)

        if len(opt_path) > 0:
            path = []
            for i in range(len(opt_path)-1):
                node1 = rrt_tree.vertices[opt_path[i]]
                node2 = rrt_tree.vertices[opt_path[i + 1]]
                node1 = (node1[0] + 4 * maze_scaling,
                         node1[1] + 4 * maze_scaling)
                node2 = (node2[0] + 4 * maze_scaling,
                         node2[1] + 4 * maze_scaling)
                path.append((node1, node2))
            lc2 = mc.LineCollection(path, colors='red', linewidths=1)
            ax.add_collection(lc2)

        if len(states) > 0:
            trace_x = []
            trace_y = []
            for i in range(len(states)):
                trace_x.append(states[i][0] + 4 * maze_scaling)
                trace_y.append(states[i][1] + 4 * maze_scaling)
            ax.plot(trace_x, trace_y, color="k", linewidth=1)

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        plt.savefig("./code/gcsl_ant/fig/rrt_star_tree" +
                    str(timestep) + "_"+str(count) + ".pdf")

    # for maze info checking [mws]
    def print_maze_infos(self):
        ant_obs = self.maze_env._get_obs()
        # ant_obs and shape
        print(ant_obs.shape)
        print(dir(self.maze_env))
        print(self.maze_env.observation_space)
        print(self.maze_env.action_space)
        # print(self.maze_env.MAZE_STRUCTURE)
        # print(self.maze_env.MAZE_HEIGHT)
        print(self.maze_env._find_robot())
        print(self.maze_env._init_positions)
