# wrapper to transplant SpecAntFall in pi-HRL to GCSL [MWS]

from argparse import _get_action_name
from gcsl.envs.goal_env import GoalEnv
from collections import OrderedDict
from gcsl.envs.ant.ant_maze_env import AntMazeEnv
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
from RRT_star.utils import Line, Box_obstacle
import math
import matplotlib.patches as patches
from RRT_star.utils import *


class AntUMazeEnv(GoalEnv, Serializable):
    def __init__(self, silent=True):
        cls = AntMazeEnv

        maze_id = 'UMaze'
        n_bins = 0
        observe_blocks = False
        put_spin_near_agent = False
        top_down_view = False
        manual_collision = False
        maze_size_scaling = 8

        gym_mujoco_kwargs = {
            'maze_id': maze_id,
            'n_bins': n_bins,
            'observe_blocks': observe_blocks,
            'put_spin_near_agent': put_spin_near_agent,
            'top_down_view': top_down_view,
            'manual_collision': manual_collision,
            'maze_size_scaling': maze_size_scaling
        }

        self.maze_env = cls(**gym_mujoco_kwargs)  # wrapped_env
        self.maze_env.reset()

        self.maze_structure = self.maze_env.MAZE_STRUCTURE
        # self.maze_structure = [
        #     [1,   1,   1,   1,   1], (x)
        #     [1,   'r', 0,   0,   1],
        #     [1,   1,   1,   0,   1],
        #     [1,   0,   0,   0,   1],
        #     [1,   1,   1,   1,   1],
        #     (y)
        # ]
        self.quick_init(locals())

        # all primitive policies
        # up = along positive x-axis; down = along negative x-axis; left = along positive y-axis; right = along negative y-axis
        self.primitives = []
        for direction in ['up', 'down', 'left', 'right']:
            # filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
            filename = '/root/code/gcsl_ant/primitives/ant_low_torque/' + direction + '.pt'
            self.primitives.append(torch.load(filename))
            # make it corresponding to the actual xy-axis
        self.low_torque_limit = torch.ones(self.lowlevel_action_space.shape[0])*30

        self.ant_radius = 0.75
        self.wall_collision_buffer = 0.25
        self.obstacles = []

        # obstacles region: -4<= x <=12, 4<= y <=12
        central_box = Box_obstacle([-4, 12], [12, 12], [-4, 4], [12, 4], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(central_box)

        left_border = Box_obstacle([-12, 28], [-4, 28], [-12, -12], [-4, -12], self.ant_radius + self.wall_collision_buffer)
        right_border = Box_obstacle([20, 28], [28, 28], [20, -12], [28, -12], self.ant_radius + self.wall_collision_buffer)
        upper_border = Box_obstacle([-4, -4], [28, -4], [-4, -12], [28, -12], self.ant_radius + self.wall_collision_buffer)
        bottom_border = Box_obstacle([-12, 28], [28, 28], [-12, 20], [28, 20], self.ant_radius + self.wall_collision_buffer)
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
        return spaces.Box(low=np.array([-4, -4]), high=np.array([20, 20]), dtype=np.float32)

    # actaully, the data it returns is tof the same shape of state, but just contain the final goal information
    def sample_goal(self):
        shape = self.observation_space.shape
        goal = np.zeros(shape)

        # do sample until goal is a feasible one
        goal_candidate = self.goal_space.sample()
        goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                         int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        while (self.maze_structure[goal_maze_idx[0] + 1][goal_maze_idx[1] + 1] == 1
               or self.goal_in_obstacle([goal_candidate[0], goal_candidate[1]]) == True):
            goal_candidate = self.goal_space.sample()
            goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                             int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]

        goal[:2] = goal_candidate
        return goal

    def goal_in_obstacle(self, goal):
        for box_obst in self.obstacles:
            if box_obst.within_box(goal) == True:
                return True
        return False

    def intersect_with_obstacle(self, line):
        if line.dirn[0] == 0:  # x = const
            if line.p0[0] > -4 and line.p0[0] < 12:
                t1 = (4 - line.p0[1])/line.dirn[1]
                t2 = (12 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            else:
                return False
        elif line.dirn[1] == 0:  # y = const
            if line.p0[1] > 4 and line.p0[1] < 12:
                t1 = (-4 - line.p0[0])/line.dirn[0]
                t2 = (12 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            else:
                return False
        else:
            for box_obst in self.obstacles:
                if box_obst.intersect_with_box(line) == True:
                    return True
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

    # param "high_action" representing distribution of primitives
    def step(self, highlevel_action):  # (highlevel_action from policy strictly means: up is go up, down is go down, etc)
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
        normalized_highlevel_action = np.array([highlevel_action[3], highlevel_action[2], highlevel_action[1], highlevel_action[0]], dtype=np.float32)
        sum = np.sum(normalized_highlevel_action)
        if sum > 0:
            normalized_highlevel_action = normalized_highlevel_action/float(sum)
        self.weights.append(normalized_highlevel_action)

        action = np.zeros(self.lowlevel_action_space.shape, dtype=np.float32)
        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            # f.write("primitive action:")
            for i in range(len(normalized_highlevel_action)):
                primitive_input = self.current_state[range(2, 29)]
                action_ = normalized_highlevel_action[i] * self.get_action_from_primitive(self.primitives[i], primitive_input[np.newaxis, :]).numpy().ravel()
                # f.write(str(self.get_action_from_primitive(self.primitives[i], primitive_input).numpy()))
                action += action_
        # action = np.clip(action, self.low_torque_limit*-1, self.low_torque_limit)
        self.actions.append(action)

        ob, reward, done, info = self.maze_env.step(action)

        self.previous_state = self.current_state
        self.current_state = ob
        done = False
        if (self.goal_in_obstacle(self.previous_state[:2]) == False) and (
                self.goal_in_obstacle(self.current_state[:2]) == False) and (
                self.intersect_with_obstacle(Line(self.previous_state[:2], self.current_state[:2])) == False):
            info['expandable'] = True
        else:
            info['expandable'] = False
        # distance = np.linalg.norm(ob[:2] - np.array([0, 16]))
        # reward = -distance
        # done = True if distance < 1.0 else False
        # progress = (1 - (distance - 1) / (16 - 1)) * 100

        # info['finished'] = False
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
        return state[..., :2]

    ###
    def _extract_sgoal(self, state):
        return state[..., :2]

    def goal_distance(self, state, goal_state):
        # self.goal_metric = 'euclidean':
        diff = self.extract_goal(state)-self.extract_goal(goal_state)
        return np.linalg.norm(diff, axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states, success_vec=[]):
        """
        Gets things to log
        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]
        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """

        for i in range(len(trajectories)):
            x = []
            y = []
            for j in range(len(trajectories[i])):
                x.append(trajectories[i][j][0] + 1 * self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 1 * self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][:2], True)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace"+str(i)+".pdf")
            else:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][:2], False)
                plt.savefig("./code/gcsl_ant/fig/trace"+str(i)+".pdf")

        return OrderedDict([])

    def plot_trajectory_fig(self, x, y, desired_goal, reached=False):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        # Draw goal
        ax.scatter(desired_goal[0] + 1 * maze_scaling, desired_goal[1] + 1 * maze_scaling, s=400, marker='*')

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
            ax.scatter(x[i] + 1 * maze_scaling, y[i] + 1 * maze_scaling, s=10, marker='*')

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, self.maze_env.MAZE_SIZE_SCALING)

        plt.savefig("./code/gcsl_ant/fig/" + self.maze_env._maze_id + "sample_goal_scatter_"+str(self.maze_env._maze_id)+".pdf")

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, rrt_tree, opt_path=[], states=[], count=-1):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        px = [x + 1 * maze_scaling for x, _ in rrt_tree.vertices]
        py = [y + 1 * maze_scaling for _, y in rrt_tree.vertices]
        fig, ax = plt.subplots()
        ax.scatter(px, py, c='cyan')
        ax.scatter([x + 1 * maze_scaling for x in goal_pos_x[-goal_scatter_num:]], [y + 1 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*')

        lines = []
        for edge in rrt_tree.edges:
            node1 = rrt_tree.vertices[edge[0]]
            node2 = rrt_tree.vertices[edge[1]]
            node1 = (node1[0] + 1 * maze_scaling, node1[1] + 1 * maze_scaling)
            node2 = (node2[0] + 1 * maze_scaling, node2[1] + 1 * maze_scaling)
            lines.append((node1, node2))
        lc = mc.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)

        if len(opt_path) > 0:
            path = []
            for i in range(len(opt_path)-1):
                node1 = rrt_tree.vertices[opt_path[i]]
                node2 = rrt_tree.vertices[opt_path[i + 1]]
                node1 = (node1[0] + 1 * maze_scaling, node1[1] + 1 * maze_scaling)
                node2 = (node2[0] + 1 * maze_scaling, node2[1] + 1 * maze_scaling)
                path.append((node1, node2))
            lc2 = mc.LineCollection(path, colors='red', linewidths=1)
            ax.add_collection(lc2)

        if len(states) > 0:
            trace_x = []
            trace_y = []
            for i in range(len(states)):
                trace_x.append(states[i][0] + 1 * maze_scaling)
                trace_y.append(states[i][1] + 1 * maze_scaling)
            ax.plot(trace_x, trace_y, color="k", linewidth=1)

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        plt.savefig("./code/gcsl_ant/fig/rrt_star_tree" + str(timestep) + "_"+str(count) + ".pdf")

    # for maze info checking 
    def print_maze_infos(self):
        ant_obs = self.maze_env._get_obs()
        # ant_obs and shape
        print(ant_obs.shape)
        print(ant_obs)
        print(dir(self.maze_env))
        print(self.maze_env.observation_space)
        print(self.maze_env.action_space)
        print(self.maze_env.MAZE_STRUCTURE)
        print(self.maze_env.MAZE_HEIGHT)

        print(self.maze_env._find_robot())
        print(self.maze_env._init_positions)


class AntFallEnv(GoalEnv, Serializable):
    def __init__(self, silent=True):
        cls = AntMazeEnv

        maze_id = 'Fall'
        n_bins = 0
        observe_blocks = False
        put_spin_near_agent = False
        top_down_view = False
        manual_collision = False
        maze_size_scaling = 8

        gym_mujoco_kwargs = {
            'maze_id': maze_id,
            'n_bins': n_bins,
            'observe_blocks': observe_blocks,
            'put_spin_near_agent': put_spin_near_agent,
            'top_down_view': top_down_view,
            'manual_collision': manual_collision,
            'maze_size_scaling': maze_size_scaling
        }

        self.maze_env = cls(**gym_mujoco_kwargs)  # wrapped_env
        self.maze_env.reset()

        self.maze_structure = self.maze_env.MAZE_STRUCTURE
        # self.maze_structure = [
        #     [1,    1,       1,  1], (x)
        #     [1,  'r',       0,  1],
        #     [1,    0, Move.YZ,  1],
        #     [1,   -1,      -1,  1],
        #     [1,    0,       0,  1],
        #     [1,    1,       1,  1],
        #     (y)
        # ]
        self.quick_init(locals())

        # all primitive policies
        # up = along positive x-axis
        # down = along negative x-axis
        # left = along positive y-axis
        # right = along negative y-axis
        self.primitives = []
        for direction in ['up', 'down', 'left', 'right']:
            # filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
            filename = '/root/code/gcsl_ant/primitives/ant_low_torque/' + direction + '.pt'
            self.primitives.append(torch.load(filename))
            # make it corresponding to the actual xy-axis
        self.low_torque_limit = torch.ones(self.lowlevel_action_space.shape[0])*30

        self.ant_radius = 0.75
        self.wall_collision_buffer = 0.25
        self.obstacles = []

        unreachable_cliff = Box_obstacle([-4, 20], [4, 20], [-4, 12], [4, 12], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(unreachable_cliff)

        left_border = Box_obstacle([-12, 36], [-4, 36], [-12, -12], [-4, -12], self.ant_radius + self.wall_collision_buffer)
        right_border = Box_obstacle([12, 36], [20, 36], [12, -12], [20, -12], self.ant_radius + self.wall_collision_buffer)
        upper_border = Box_obstacle([-12, -4], [20, -4], [-12, -12], [20, -12], self.ant_radius + self.wall_collision_buffer)
        bottom_border = Box_obstacle([-12, 36], [20, 36], [-12, 28], [20, 28], self.ant_radius + self.wall_collision_buffer)
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
    # observation space here should be np.concatenate([ant_obs( return of step(action) ), box_obs( movable block pos )])
    def observation_space(self):
        ant_obs = self.maze_env._get_obs()
        box_obs = self.maze_env.wrapped_env.data.get_body_xpos('movable_2_2')
        shape = np.concatenate([ant_obs, box_obs]).shape
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
        return spaces.Box(low=np.array([-4, -4]), high=np.array([12, 28]), dtype=np.float32)

    # actaully, the data it returns is tof the same shape of state, but just contain the final goal information
    def sample_goal(self):
        shape = self.observation_space.shape
        goal = np.zeros(shape)

        goal_candidate = self.goal_space.sample()
        goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                         int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        while (self.maze_structure[goal_maze_idx[0] + 1][goal_maze_idx[1] + 1] == 1
               or self.goal_in_obstacle([goal_candidate[0], goal_candidate[1]]) == True):
            goal_candidate = self.goal_space.sample()
            goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                             int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]

        goal[:2] = goal_candidate
        return goal

    def goal_in_obstacle(self, goal):
        for box_obst in self.obstacles:
            if box_obst.within_box(goal) == True:
                return True
        return False

    def intersect_with_obstacle(self, line):
        # TODO: IF NEEDED, obstacles region: -4<= x <=4, 12<= y <=20

        return False

    ###
    def reset(self):
        """
        Resets the environment and returns a state vector

        Returns:
            The initial state
        """
        ant_obs = self.maze_env.reset()
        box_obs = self.maze_env.wrapped_env.data.get_body_xpos('movable_2_2')
        obs = np.concatenate([ant_obs, box_obs])
        self.current_state = obs
        self.previous_state = self.current_state
        self.weights = []
        self.actions = []
        return obs

    def step(self, highlevel_action):  # (highlevel_action)
        """
        Runs 1 step of simulation

        Returns:
            A tuple containing:
                next_state
                reward (always 0)
                done
                infos
        """
        normalized_highlevel_action = np.array([highlevel_action[3], highlevel_action[2], highlevel_action[1], highlevel_action[0]], dtype=np.float32)
        sum = np.sum(normalized_highlevel_action)
        if sum > 0:
            normalized_highlevel_action = normalized_highlevel_action/float(sum)
        self.weights.append(normalized_highlevel_action)

        normalized_highlevel_action = normalized_highlevel_action

        action = np.zeros(self.lowlevel_action_space.shape, dtype=np.float32)
        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            # f.write("primitive action:")
            for i in range(len(normalized_highlevel_action)):
                primitive_input = self.current_state[range(2, 29)]
                # In AntFall environment, the whole environment is elevated by a box with height of 4.
                primitive_input[0] -= 4
                action_ = normalized_highlevel_action[i] * self.get_action_from_primitive(self.primitives[i], primitive_input[np.newaxis, :]).numpy().ravel()
                # f.write(str(self.get_action_from_primitive(self.primitives[i], primitive_input).numpy()))
                action += action_
        self.actions.append(action)

        ob, reward, done, info = self.maze_env.step(action)

        box_ob = self.maze_env.wrapped_env.data.get_body_xpos('movable_2_2')

        self.previous_state = self.current_state
        ob = np.concatenate([ob, box_ob])
        self.current_state = ob
        done = False
        posx, posy, posz = self.current_state[:3]
        if posx > -4 and posx < 12 and posy > 12 and posy < 20 and posz < 1:  # range [-4, 12] × [12, 20]
            # with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            #     f.write(str(self.current_state[:3])+ str(self.current_state[-3:])  + "\r")
            #     f.write("ANT FALL INTO CLIFF\r")
            # f.close()
            info['expandable'] = False
        else:
            if (self.goal_in_obstacle(self.previous_state[:2]) == False) and (
                    self.goal_in_obstacle(self.current_state[:2]) == False) and (
                    self.intersect_with_obstacle(Line(self.previous_state[:2], self.current_state[:2])) == False):
                info['expandable'] = True
            else:
                info['expandable'] = False
        # distance = np.linalg.norm(ob[:2] - np.array([0, 27]))
        # reward = -distance

        # # if ant has fallen down so that can't move

        # progress = (1 - (distance - 1) / (27 - 1)) * 100
        # info['finished'] = False
        # info['distance'] = (distance - 1) / (27 - 1) if distance > 1.0 else 0.0
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
        return state[..., :2]

    ###
    def _extract_sgoal(self, state):
        return state[..., :2]

    def goal_distance(self, state, goal_state):
        # self.goal_metric = 'euclidean':
        diff = self.extract_goal(state)-self.extract_goal(goal_state)
        return np.linalg.norm(diff, axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states, success_vec=[]):
        """
        Gets things to log

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """

        for i in range(len(trajectories)):
            x = []
            y = []
            for j in range(len(trajectories[i])):
                x.append(trajectories[i][j][0] + 1 * self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 1 * self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0:2], True)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace"+str(i)+".pdf")
            else:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0:2], False)
                plt.savefig("./code/gcsl_ant/fig/trace"+str(i)+".pdf")

        return OrderedDict([])

    def plot_trajectory_fig(self, x, y, desired_goal, reached=False):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        # Draw goal
        ax.scatter(desired_goal[0] + 1 * maze_scaling, desired_goal[1] + 1 * maze_scaling, s=400, marker='*')

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
            ax.scatter(x[i] + 1 * maze_scaling, y[i] + 1 * maze_scaling, s=10, marker='*')

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, self.maze_env.MAZE_SIZE_SCALING)

        plt.savefig("./code/gcsl_ant/fig/" + self.maze_env._maze_id + "sample_goal_scatter_"+str(self.maze_env._maze_id)+".pdf")

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, rrt_tree, opt_path=[], states=[], count=-1):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        px = [x + 1 * maze_scaling for x, _ in rrt_tree.vertices]
        py = [y + 1 * maze_scaling for _, y in rrt_tree.vertices]
        fig, ax = plt.subplots()
        ax.scatter(px, py, c='cyan')
        ax.scatter([x + 1 * maze_scaling for x in goal_pos_x[-goal_scatter_num:]], [y + 1 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*')

        lines = []
        for edge in rrt_tree.edges:
            node1 = rrt_tree.vertices[edge[0]]
            node2 = rrt_tree.vertices[edge[1]]
            node1 = (node1[0] + 1 * maze_scaling, node1[1] + 1 * maze_scaling)
            node2 = (node2[0] + 1 * maze_scaling, node2[1] + 1 * maze_scaling)
            lines.append((node1, node2))
        lc = mc.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)

        if len(opt_path) > 0:
            path = []
            for i in range(len(opt_path)-1):
                node1 = rrt_tree.vertices[opt_path[i]]
                node2 = rrt_tree.vertices[opt_path[i + 1]]
                node1 = (node1[0] + 1 * maze_scaling, node1[1] + 1 * maze_scaling)
                node2 = (node2[0] + 1 * maze_scaling, node2[1] + 1 * maze_scaling)
                path.append((node1, node2))
            lc2 = mc.LineCollection(path, colors='red', linewidths=1)
            ax.add_collection(lc2)

        if len(states) > 0:
            trace_x = []
            trace_y = []
            for i in range(len(states)):
                trace_x.append(states[i][0] + 1 * maze_scaling)
                trace_y.append(states[i][1] + 1 * maze_scaling)
            ax.plot(trace_x, trace_y, color="k", linewidth=1)

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        plt.savefig("./code/gcsl_ant/fig/rrt_star_tree" + str(timestep) + "_"+str(count) + ".pdf")

    # for maze info checking [mws]
    def print_maze_infos(self):
        ant_obs = self.maze_env._get_obs()
        box_obs = self.maze_env.wrapped_env.data.get_body_xpos('movable_2_2')
        # ant_obs shape
        print(ant_obs.shape)
        print(ant_obs)
        # box_obs shape:
        print(box_obs.shape)
        print(box_obs)
        print(dir(self.maze_env))
        print(self.maze_env.observation_space)
        print(self.maze_env.action_space)
        print(self.maze_env.MAZE_STRUCTURE)
        print(self.maze_env.MAZE_HEIGHT)

        print(self.maze_env._find_robot())
        print(self.maze_env._init_positions)
        print(self.maze_env._init_torso_x)
        print(self.maze_env._init_torso_y)
        print(self.maze_env.movable_blocks)
        print(self.maze_env.get_range_sensor_obs())


class AntPushEnv(GoalEnv, Serializable):
    def __init__(self, silent=True):
        cls = AntMazeEnv

        maze_id = 'Push'
        n_bins = 0
        observe_blocks = False
        put_spin_near_agent = False
        top_down_view = False
        manual_collision = False
        maze_size_scaling = 8

        gym_mujoco_kwargs = {
            'maze_id': maze_id,
            'n_bins': n_bins,
            'observe_blocks': observe_blocks,
            'put_spin_near_agent': put_spin_near_agent,
            'top_down_view': top_down_view,
            'manual_collision': manual_collision,
            'maze_size_scaling': maze_size_scaling
        }

        self.maze_env = cls(**gym_mujoco_kwargs)  # wrapped_env
        self.maze_env.reset()

        self.maze_structure = self.maze_env.MAZE_STRUCTURE
        # self.maze_structure = [
        #     [1, 1,        1, 1, 1], (x)
        #     [1, 0,      'r', 1, 1],
        #     [1, 0,  Move.XY, 0, 1],
        #     [1, 1,        0, 1, 1],
        #     [1, 1,        1, 1, 1],
        #     (y)
        # ]

        self.quick_init(locals())

        # all primitive policies
        # up = along positive x-axis
        # down = along negative x-axis
        # left = along positive y-axis
        # right = along negative y-axis
        self.primitives = []
        for direction in ['up', 'down', 'left', 'right']:
            # filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
            filename = '/root/code/gcsl_ant/primitives/ant_low_torque/' + direction + '.pt'
            self.primitives.append(torch.load(filename))
            # make it corresponding to the actual xy-axis
        self.low_torque_limit = torch.ones(self.lowlevel_action_space.shape[0]) * 30

        self.ant_radius = 0.75
        self.wall_collision_buffer = 0.25
        self.obstacles = []

        # obstacles region:
        # 1) 4<= x <=12, -4<= y <=4
        # 2) -12<= x <=-4, 12<= y <=20
        # 3) 4<= x <=12, 12<= y <=20
        box_topright = Box_obstacle([4, 4], [12, 4], [4, -4], [12, -4], self.ant_radius + self.wall_collision_buffer)
        box_bottomleft = Box_obstacle([-12, 20], [-4, 20], [-12, 12], [-4, 12], self.ant_radius + self.wall_collision_buffer)
        box_bottomright = Box_obstacle([4, 20], [12, 20], [4, 12], [12, 12], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(box_topright)
        self.obstacles.append(box_bottomleft)
        self.obstacles.append(box_bottomright)

        left_border = Box_obstacle([-20, 28], [-12, 28], [-20, -12], [-12, -12], self.ant_radius + self.wall_collision_buffer)
        right_border = Box_obstacle([12, 28], [20, 28], [12, -12], [20, -12], self.ant_radius + self.wall_collision_buffer)
        upper_border = Box_obstacle([-20, -4], [20, -4], [-20, -12], [20, -12], self.ant_radius + self.wall_collision_buffer)
        bottom_border = Box_obstacle([-20, 28], [20, 28], [-20, 20], [20, 20], self.ant_radius + self.wall_collision_buffer)
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
    # observation space here should be np.concatenate([ant_obs( return of step(action) ), box_obs( movable block pos )])
    def observation_space(self):
        shape = self.maze_env._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        return spaces.Box(low, high, dtype=np.float32)

    # action space is a 4-dim vector which indicate four primitives
    @property
    def action_space(self):
        low = np.zeros((4))
        high = np.ones((4))
        return spaces.Box(low, high, dtype=np.float32)

    @property
    def lowlevel_action_space(self):
        return self.maze_env.action_space

    @property
    def state_space(self):
        return self.observation_space

    @property
    def goal_space(self):
        return spaces.Box(low=np.array([-12, -4]), high=np.array([12, 20]), dtype=np.float32)

    # actaully, the data it returns is tof the same shape of state, but just contain the final goal information
    def sample_goal(self):
        shape = self.observation_space.shape
        goal = np.zeros(shape)

        goal_candidate = self.goal_space.sample()
        goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                         int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        while (self.maze_structure[goal_maze_idx[0] + 1][goal_maze_idx[1] + 2] == 1
               or self.goal_in_obstacle([goal_candidate[0], goal_candidate[1]]) == True):
            goal_candidate = self.goal_space.sample()
            goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                             int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]

        goal[: 2] = goal_candidate
        return goal

    def goal_in_obstacle(self, goal):
        for box_obst in self.obstacles:
            if box_obst.within_box(goal) == True:
                return True
        return False

    def intersect_with_obstacle(self, line):

        if line.dirn[0] == 0:  # x = const
            if line.p0[0] > -4 and line.p0[0] < 4:
                return False
            elif line.p0[0] > -12 and line.p0[0] < -4:
                t1 = (12 - line.p0[1])/line.dirn[1]
                t2 = (20 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            else:  # line.p0[0] > 4 and line.p0[0] < 12
                t1 = (4 - line.p0[1])/line.dirn[1]
                t2 = (12 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 > line.dist) or (t1 > line.dist and t2 < 0):
                    return False
                else:
                    return True
        elif line.dirn[1] == 0:  # y = const
            if line.p0[1] > 4 and line.p0[1] < 12:
                return False
            elif line.p0[1] > -4 and line.p0[1] < 4:
                t1 = (4 - line.p0[0])/line.dirn[0]
                t2 = (12 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            else:  # line.p0[1] > 12 and line.p0[1] < 20:
                t1 = (-4 - line.p0[0])/line.dirn[0]
                t2 = (4 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 > line.dist) or (t1 > line.dist and t2 < 0):
                    return False
                else:
                    return True
        else:
            for box_obst in self.obstacles:
                if box_obst.intersect_with_box(line) == True:
                    return True
            return False

    ###
    def reset(self):
        """
        Resets the environment and returns a state vector

        Returns:
            The initial state
        """
        ant_obs = self.maze_env.reset()
        self.current_state = ant_obs
        self.previous_state = self.current_state
        self.weights = []
        self.actions = []
        return ant_obs

    def step(self, highlevel_action):  # (highlevel_action)
        """
        Runs 1 step of simulation

        Returns:
            A tuple containing:
                next_state
                reward (always 0)
                done
                infos
        """
        normalized_highlevel_action = np.array([highlevel_action[3], highlevel_action[2], highlevel_action[1], highlevel_action[0]], dtype=np.float32)
        sum = np.sum(normalized_highlevel_action)
        if sum > 0:
            normalized_highlevel_action = normalized_highlevel_action/float(sum)
        self.weights.append(normalized_highlevel_action)

        action = np.zeros(self.lowlevel_action_space.shape, dtype=np.float32)
        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            # f.write("primitive action:")
            for i in range(len(normalized_highlevel_action)):
                primitive_input = self.current_state[range(2, 29)]
                action_ = normalized_highlevel_action[i] * self.get_action_from_primitive(self.primitives[i], primitive_input[np.newaxis, :]).numpy().ravel()
                # f.write(str(self.get_action_from_primitive(self.primitives[i], primitive_input).numpy()))
                action += action_
        self.actions.append(action)

        ob, reward, _, info = self.maze_env.step(action)

        self.previous_state = self.current_state
        self.current_state = ob
        done = False
        if (self.goal_in_obstacle(self.previous_state[:2]) == False) and (
                self.goal_in_obstacle(self.current_state[:2]) == False) and (
                self.intersect_with_obstacle(Line(self.previous_state[:2], self.current_state[:2])) == False):
            info['expandable'] = True
        else:
            info['expandable'] = False
        # progress = (1 - (distance - 1) / (27 - 1)) * 100
        # info['finished'] = False
        # info['distance'] = (distance - 1) / (27 - 1) if distance > 1.0 else 0.0
        # info['progress'] = progress
        return ob, reward, done, info

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
        return state[..., :2]

    ###
    def _extract_sgoal(self, state):
        return state[..., :2]

    def goal_distance(self, state, goal_state):
        # self.goal_metric = 'euclidean':
        diff = self.extract_goal(state) - self.extract_goal(goal_state)
        return np.linalg.norm(diff, axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states, success_vec=[]):
        """
        Gets things to log

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """

        for i in range(len(trajectories)):
            x = []
            y = []
            for j in range(len(trajectories[i])):
                x.append(trajectories[i][j][0] + 2 * self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 1 * self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][:2], True)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace"+str(i)+".pdf")
            else:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][:2], False)
                plt.savefig("./code/gcsl_ant/fig/trace"+str(i)+".pdf")

        return OrderedDict([])

    def plot_trajectory_fig(self, x, y, desired_goal, reached=False):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        # Draw goal
        ax.scatter(desired_goal[0] + 2 * maze_scaling, desired_goal[1] + 1 * maze_scaling, s=400, marker='*')

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
            ax.scatter(x[i] + 2 * maze_scaling, y[i] + 1 * maze_scaling, s=10, marker='*')

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, self.maze_env.MAZE_SIZE_SCALING)

        plt.savefig("./code/gcsl_ant/fig/" + self.maze_env._maze_id + "sample_goal_scatter_"+str(self.maze_env._maze_id)+".pdf")

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, rrt_tree, opt_path=[], states=[], count=-1):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING

        px = [x + 2 * maze_scaling for x, _ in rrt_tree.vertices]
        py = [y + 1 * maze_scaling for _, y in rrt_tree.vertices]
        fig, ax = plt.subplots()
        ax.scatter(px, py, c='cyan', s=10)
        ax.scatter([x + 2 * maze_scaling for x in goal_pos_x[-goal_scatter_num:]], [y + 1 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*')

        lines = []
        for edge in rrt_tree.edges:
            node1 = rrt_tree.vertices[edge[0]]
            node2 = rrt_tree.vertices[edge[1]]
            node1 = (node1[0] + 2 * maze_scaling, node1[1] + 1 * maze_scaling)
            node2 = (node2[0] + 2 * maze_scaling, node2[1] + 1 * maze_scaling)
            lines.append((node1, node2))
        lc = mc.LineCollection(lines, colors='green', linewidths=1)
        ax.add_collection(lc)

        if len(opt_path) > 0:
            path = []
            for i in range(len(opt_path)-1):
                node1 = rrt_tree.vertices[opt_path[i]]
                node2 = rrt_tree.vertices[opt_path[i + 1]]
                node1 = (node1[0] + 2 * maze_scaling, node1[1] + 1 * maze_scaling)
                node2 = (node2[0] + 2 * maze_scaling, node2[1] + 1 * maze_scaling)
                path.append((node1, node2))
            lc2 = mc.LineCollection(path, colors='red', linewidths=1)
            ax.add_collection(lc2)

        if len(states) > 0:
            trace_x = []
            trace_y = []
            for i in range(len(states)):
                trace_x.append(states[i][0] + 2 * maze_scaling)
                trace_y.append(states[i][1] + 1 * maze_scaling)
            ax.plot(trace_x, trace_y, color="k", linewidth=1)

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        plt.savefig("./code/gcsl_ant/fig/rrt_star_tree" + str(timestep) + "_"+str(count) + ".pdf")

    # for maze info checking [mws]
    def print_maze_infos(self):
        ant_obs = self.maze_env._get_obs()
        # ant_obs shape
        print(ant_obs.shape)
        print(ant_obs)
        # box_obs shape:
        print(dir(self.maze_env))
        print(self.observation_space)
        print(self.action_space)
        print(self.action_space.shape)
        print(self.maze_env.MAZE_STRUCTURE)
        print(self.maze_env.MAZE_HEIGHT)

        print(self.maze_env._find_robot())
        print(self.maze_env._init_positions)
        print(self.maze_env.movable_blocks)
        print(self.maze_env.get_range_sensor_obs())
