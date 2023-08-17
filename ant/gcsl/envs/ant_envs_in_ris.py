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
import torch
from dependencies.multiworld.core.serializable import Serializable
from PIL import Image
import torch
import torch.nn as nn
from RRT_star.utils import Line, Box_obstacle
import math
import matplotlib.patches as patches
from RRT_star.utils import *


class AntLongUMazeEnv(GoalEnv, Serializable):
    def __init__(self, silent=True):
        cls = AntMazeEnv

        maze_id = 'AntLongU'
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

        self.maze_structure = self.maze_env.MAZE_STRUCTURE
        # structure = [
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0,'r',0, 1, 1, 0, 0, 0, 0, 1], (x)
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #           (y)
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

        # obstacles region:
        # when 1.5<= x <=3.5, -2.5<= y <=17.5
        self.ant_radius = 0.75
        self.wall_collision_buffer = 0.25
        self.obstacles = []
        box_center = Box_obstacle([1.5, 17.5], [3.5, 17.5], [1.5, -2.5], [3.5, -2.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(box_center)

        left_border = Box_obstacle([-3.5, 22.5], [-2.5, 22.5], [-3.5, -3.5], [-2.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        right_border = Box_obstacle([7.5, 22.5], [8.5, 22.5], [7.5, -3.5], [8.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        upper_border = Box_obstacle([-3.5, -2.5], [8.5, -2.5], [-3.5, -3.5], [8.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        bottom_border = Box_obstacle([-3.5, 22.5], [8.5, 22.5], [-3.5, 21.5], [8.5, 21.5], self.ant_radius + self.wall_collision_buffer)
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
        return spaces.Box(low=np.array([-2, -2]), high=np.array([7, 21]), dtype=np.float32)

    # actaully, the data it returns is tof the same shape of state, but just contain the final goal information
    def sample_goal(self):
        shape = self.observation_space.shape
        goal = np.zeros(shape)

        # do sample until goal is a feasible one
        goal_candidate = self.goal_space.sample()
        goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                         int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        while (self.maze_structure[goal_maze_idx[0] + 3][goal_maze_idx[1] + 3] == 1
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

        if line.dirn[0] == 0:  # x = const
            if line.p0[0] > 1.5 and line.p0[0] < 3.5:
                t1 = (-2.5 - line.p0[1])/line.dirn[1]
                t2 = (17.5 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            else:
                return False
        elif line.dirn[1] == 0:  # y = const
            if line.p0[1] > -2.5 and line.p0[1] < 17.5:
                t1 = (1.5 - line.p0[0])/line.dirn[0]
                t2 = (3.5 - line.p0[0])/line.dirn[0]
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

    def get_diagnostics(self, trajectories, desired_goal_states, success_vec=[]):
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
                x.append(trajectories[i][j][0] + 3 * self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 3 * self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0: 2], True)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace"+str(i)+".pdf")
            else:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0: 2], False)
                plt.savefig("./code/gcsl_ant/fig/trace"+str(i)+".pdf")

        return OrderedDict([])

    def plot_trajectory_fig(self, x, y, desired_goal, reached=False):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        # Draw goal
        ax.scatter(desired_goal[0] + 3 * maze_scaling, desired_goal[1] + 3 * maze_scaling, s=400, marker='*')

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
            ax.scatter(x[i] + 3 * maze_scaling, y[i] + 3 * maze_scaling, s=10, marker='*')

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, self.maze_env.MAZE_SIZE_SCALING)

        plt.savefig("./code/gcsl_ant/fig/" + self.maze_env._maze_id + "sample_goal_scatter_"+str(self.maze_env._maze_id)+".pdf")

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, rrt_tree, opt_path=[], states=[], count=-1):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        px = [x + 3 * maze_scaling for x, y in rrt_tree.vertices]
        py = [y + 3 * maze_scaling for x, y in rrt_tree.vertices]
        fig, ax = plt.subplots()
        ax.scatter(px, py, c='cyan')
        ax.scatter([x + 3 * maze_scaling for x in goal_pos_x[-goal_scatter_num:]], [y + 3 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*')

        lines = []
        for edge in rrt_tree.edges:
            node1 = rrt_tree.vertices[edge[0]]
            node2 = rrt_tree.vertices[edge[1]]
            node1 = (node1[0] + 3 * maze_scaling, node1[1] + 3 * maze_scaling)
            node2 = (node2[0] + 3 * maze_scaling, node2[1] + 3 * maze_scaling)
            lines.append((node1, node2))
        lc = mc.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)
        
        if len(opt_path) > 0:
            path = []
            for i in range(len(opt_path)-1):
                node1 = rrt_tree.vertices[opt_path[i]]
                node2 = rrt_tree.vertices[opt_path[i + 1]]
                node1 = (node1[0] + 3 * maze_scaling, node1[1] + 3 * maze_scaling)
                node2 = (node2[0] + 3 * maze_scaling, node2[1] + 3 * maze_scaling)
                path.append((node1, node2))
            lc2 = mc.LineCollection(path, colors='red', linewidths=1)
            ax.add_collection(lc2)

        if len(states) > 0:
            trace_x = []
            trace_y = []
            for i in range(len(states)):
                trace_x.append(states[i][0] + 3 * maze_scaling)
                trace_y.append(states[i][1] + 3 * maze_scaling)
            ax.plot(trace_x, trace_y, color="k", linewidth=1)

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        plt.savefig("./code/gcsl_ant/fig/rrt_star_tree" + str(timestep) + "_"+str(count) + ".pdf")

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


class AntSMazeEnv(GoalEnv, Serializable):
    def __init__(self, silent=True):
        cls = AntMazeEnv

        maze_id = 'AntS'
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

        self.maze_structure = self.maze_env.MAZE_STRUCTURE
        # structure = [
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0,'r',0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],(x)
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #              (y)
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

        # obstacles region:
        # when 1.5<= x <=3.5, -2.5<= y <=9.5
        #      7.5<= x <=9.5, 1.5<= y <=13.5
        self.ant_radius = 0.75
        self.wall_collision_buffer = 0.25
        self.obstacles = []
        box_topleft = Box_obstacle([1.5, 9.5], [3.5, 9.5], [1.5, -2.5], [3.5, -2.5], self.ant_radius + self.wall_collision_buffer)
        box_bottomright = Box_obstacle([7.5, 13.5], [9.5, 13.5], [7.5, 1.5], [9.5, 1.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(box_topleft)
        self.obstacles.append(box_bottomright)

        left_border = Box_obstacle([-3.5, 14.5], [-2.5, 14.5], [-3.5, -3.5], [-2.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        right_border = Box_obstacle([13.5, 14.5], [14.5, 14.5], [13.5, -3.5], [14.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        upper_border = Box_obstacle([-3.5, -2.5], [14.5, -2.5], [-3.5, -3.5], [14.5, -3.5], self.ant_radius + self.wall_collision_buffer)
        bottom_border = Box_obstacle([-3.5, 14.5], [14.5, 14.5], [-3.5, 13.5], [14.5, 13.5], self.ant_radius + self.wall_collision_buffer)
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
        return spaces.Box(low=np.array([-2, -2]), high=np.array([13, 13]), dtype=np.float32)

    # actaully, the data it returns is tof the same shape of state, but just contain the final goal information
    def sample_goal(self):
        shape = self.observation_space.shape
        goal = np.zeros(shape)

        # do sample until goal is a feasible one
        goal_candidate = self.goal_space.sample()
        goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                         int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        while (self.maze_structure[goal_maze_idx[0] + 3][goal_maze_idx[1] + 3] == 1
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
            if line.p0[0] > 1.5 and line.p0[0] < 3.5:
                t1 = (-2.5 - line.p0[1])/line.dirn[1]
                t2 = (9.5 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            elif line.p0[0] > 7.5 and line.p0[0] < 9.5:
                t1 = (1.5 - line.p0[1])/line.dirn[1]
                t2 = (13.5 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            else:
                return False
        elif line.dirn[1] == 0:  # y = const
            intersect_with_topleft = False
            intersect_with_bottomright = False
            if line.p0[1] > -2.5 and line.p0[1] < 9.5:
                t1 = (1.5 - line.p0[0])/line.dirn[0]
                t2 = (3.5 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    intersect_with_topleft = False
                else:
                    intersect_with_topleft = True

            if line.p0[1] > 1.5 and line.p0[1] < 13.5:
                t1 = (7.5 - line.p0[0])/line.dirn[0]
                t2 = (9.5 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    intersect_with_bottomright = False
                else:
                    intersect_with_bottomright = True
            if intersect_with_topleft == False and intersect_with_bottomright == False:
                return False
            else:
                return True
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
        
        # if (self.goal_in_obstacle(self.previous_state[:2]) == False) and (
        #         self.goal_in_obstacle(self.current_state[:2]) == False) and (
        #         self.intersect_with_obstacle(Line(self.previous_state[:2], self.current_state[:2])) == False):
        #     info['expandable'] = True
        # else:
        #     info['expandable'] = False

        info['expandable'] = True
        
        # if self.goal_in_obstacle(self.current_state):
        #     info['halt'] = True
        # else:
        #     info['halt'] = False

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
        return state[..., : 2]

    ###
    def _extract_sgoal(self, state):
        return state[..., : 2]

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
                x.append(trajectories[i][j][0] + 3 * self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 3 * self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0: 2], True)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace"+str(i)+".pdf")
            else:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0: 2], False)
                plt.savefig("./code/gcsl_ant/fig/trace"+str(i)+".pdf")

        return OrderedDict([])

    def plot_trajectory_fig(self, x, y, desired_goal, reached=False):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        # Draw goal
        ax.scatter(desired_goal[0] + 3 * maze_scaling, desired_goal[1] + 3 * maze_scaling, s=400, marker='*')

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
            ax.scatter(x[i] + 3 * maze_scaling, y[i] + 3 * maze_scaling, s=10, marker='*')

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, self.maze_env.MAZE_SIZE_SCALING)

        plt.savefig("./code/gcsl_ant/fig/" + self.maze_env._maze_id + "sample_goal_scatter_"+str(self.maze_env._maze_id)+".pdf")

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, rrt_tree, opt_path=[], states=[], count=-1):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        px = [x + 3 * maze_scaling for x, y in rrt_tree.vertices]
        py = [y + 3 * maze_scaling for x, y in rrt_tree.vertices]
        fig, ax = plt.subplots()
        ax.scatter(px, py, c='cyan')
        ax.scatter([x + 3 * maze_scaling for x in goal_pos_x[-goal_scatter_num:]], [y + 3 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*')

        lines = []
        for edge in rrt_tree.edges:
            node1 = rrt_tree.vertices[edge[0]]
            node2 = rrt_tree.vertices[edge[1]]
            node1 = (node1[0] + 3 * maze_scaling, node1[1] + 3 * maze_scaling)
            node2 = (node2[0] + 3 * maze_scaling, node2[1] + 3 * maze_scaling)
            lines.append((node1, node2))
        lc = mc.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)
        
        if len(opt_path) > 0:
            path = []
            for i in range(len(opt_path)-1):
                node1 = rrt_tree.vertices[opt_path[i]]
                node2 = rrt_tree.vertices[opt_path[i + 1]]
                node1 = (node1[0] + 3 * maze_scaling, node1[1] + 3 * maze_scaling)
                node2 = (node2[0] + 3 * maze_scaling, node2[1] + 3 * maze_scaling)
                path.append((node1, node2))
            lc2 = mc.LineCollection(path, colors='red', linewidths=1)
            ax.add_collection(lc2)

        if len(states) > 0:
            trace_x = []
            trace_y = []
            for i in range(len(states)):
                trace_x.append(states[i][0] + 3 * maze_scaling)
                trace_y.append(states[i][1] + 3 * maze_scaling)
            ax.plot(trace_x, trace_y, color="k", linewidth=1)

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        plt.savefig("./code/gcsl_ant/fig/rrt_star_tree" + str(timestep) + "_"+str(count) + ".pdf")

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


class AntPiMazeEnv(GoalEnv, Serializable):
    def __init__(self, silent=True):
        cls = AntMazeEnv

        maze_id = 'AntPi'
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

        self.maze_structure = self.maze_env.MAZE_STRUCTURE
        # structure = [
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0,'r',0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],(x)
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],(8)
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],(9)
        #     [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                       (y)   (8)(9)
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

        # obstacles region:
        # when -5.5<= x <=-1.5 or 6.5<= x <=10.5, 7.5<= y <=9.5
        #      1.5<= x <=3.5, -1.5<= y <=10.5
        #      -1.5<= x <=6.5, 2.5<= y <=4.5
        self.ant_radius = 0.75
        self.wall_collision_buffer = 0.25
        self.obstacles = []
        box_west = Box_obstacle([-5.5, 9.5], [-1.5, 9.5], [-5.5, 7.5], [-1.5, 7.5], self.ant_radius + self.wall_collision_buffer)
        box_east = Box_obstacle([6.5, 9.5], [10.5, 9.5], [6.5, 7.5], [10.5, 7.5], self.ant_radius + self.wall_collision_buffer)
        box_vertical = Box_obstacle([1.5, 10.5], [3.5, 10.5], [1.5, -1.5], [3.5, -1.5], self.ant_radius + self.wall_collision_buffer)
        box_horizontal = Box_obstacle([-1.5, 4.5], [6.5, 4.5], [-1.5, 2.5], [6.5, 2.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(box_west)
        self.obstacles.append(box_east)
        self.obstacles.append(box_vertical)
        self.obstacles.append(box_horizontal)

        left_border = Box_obstacle([-6.5, 15.5], [-5.5, 15.5], [-6.5, -2.5], [-5.5, -2.5], self.ant_radius + self.wall_collision_buffer)
        right_border = Box_obstacle([10.5, 15.5], [11.5, 15.5], [10.5, -2.5], [11.5, -2.5], self.ant_radius + self.wall_collision_buffer)
        upper_border = Box_obstacle([-6.5, -1.5], [11.5, -1.5], [-6.5, -2.5], [11.5, -2.5], self.ant_radius + self.wall_collision_buffer)
        bottom_border = Box_obstacle([-6.5, 15.5], [11.5, 15.5], [-6.5, 14.5], [11.5, 14.5], self.ant_radius + self.wall_collision_buffer)
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
        return spaces.Box(low=np.array([-5, -1]), high=np.array([10, 14]), dtype=np.float32)

    # actaully, the data it returns is tof the same shape of state, but just contain the final goal information
    def sample_goal(self):
        shape = self.observation_space.shape
        goal = np.zeros(shape)

        # do sample until goal is a feasible one
        goal_candidate = self.goal_space.sample()
        goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                         int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        while (self.maze_structure[goal_maze_idx[0] + 2][goal_maze_idx[1] + 6] == 1
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
            if (line.p0[0] > -5.5 and line.p0[0] < -1.5) or (line.p0[0] > 6.5 and line.p0[0] < 10.5):
                t1 = (7.5 - line.p0[1])/line.dirn[1]
                t2 = (9.5 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            elif (line.p0[0] > -1.5 and line.p0[0] < 1.5) or (line.p0[0] > 3.5 and line.p0[0] < 6.5):
                t1 = (2.5 - line.p0[1])/line.dirn[1]
                t2 = (4.5 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            else:
                t1 = (-1.5 - line.p0[1])/line.dirn[1]
                t2 = (10.5 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
        elif line.dirn[1] == 0:  # y = const
            if line.p0[1] > 9.5 and line.p0[1] < 10.5:
                t1 = (1.5 - line.p0[0])/line.dirn[0]
                t2 = (3.5 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            elif line.p0[1] > 7.5 and line.p0[1] < 9.5:
                in_left_entrance = False
                in_right_entrance = False
                t1 = (-1.5 - line.p0[0])/line.dirn[0]
                t2 = (1.5 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 > line.dist) or (t1 > line.dist and t2 < 0):
                    in_left_entrance = True
                else:
                    in_left_entrance = False

                t3 = (3.5 - line.p0[0])/line.dirn[0]
                t4 = (6.5 - line.p0[0])/line.dirn[0]
                if (t3 < 0 and t4 > line.dist) or (t3 > line.dist and t4 < 0):
                    in_right_entrance = True
                else:
                    in_right_entrance = False

                if in_left_entrance or in_right_entrance:
                    return False
                else:
                    return True
            elif (line.p0[1] > 4.5 and line.p0[1] < 7.5) or (line.p0[1] > -1.5 and line.p0[1] < 2.5):
                t1 = (1.5 - line.p0[0])/line.dirn[0]
                t2 = (3.5 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            elif line.p0[1] > 2.5 and line.p0[1] < 4.5:
                t1 = (-1.5 - line.p0[0])/line.dirn[0]
                t2 = (6.5 - line.p0[0])/line.dirn[0]
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
        return state[..., : 2]

    ###
    def _extract_sgoal(self, state):
        return state[..., : 2]

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
                x.append(trajectories[i][j][0] + 6 * self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 2 * self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0: 2], True)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace"+str(i)+".pdf")
            else:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0: 2], False)
                plt.savefig("./code/gcsl_ant/fig/trace"+str(i)+".pdf")

        return OrderedDict([])

    def plot_trajectory_fig(self, x, y, desired_goal, reached=False):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        # Draw goal
        ax.scatter(desired_goal[0] + 6 * maze_scaling, desired_goal[1] + 2 * maze_scaling, s=400, marker='*')

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
            ax.scatter(x[i] + 6 * maze_scaling, y[i] + 2 * maze_scaling, s=10, marker='*')

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, self.maze_env.MAZE_SIZE_SCALING)

        plt.savefig("./code/gcsl_ant/fig/" + self.maze_env._maze_id + "sample_goal_scatter_"+str(self.maze_env._maze_id)+".pdf")

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, rrt_tree, opt_path=[], states=[], count=-1):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        px = [x + 6 * maze_scaling for x, y in rrt_tree.vertices]
        py = [y + 2 * maze_scaling for x, y in rrt_tree.vertices]
        fig, ax = plt.subplots()
        ax.scatter(px, py, c='cyan')
        ax.scatter([x + 6 * maze_scaling for x in goal_pos_x[-goal_scatter_num:]], [y + 2 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*')

        lines = []
        for edge in rrt_tree.edges:
            node1 = rrt_tree.vertices[edge[0]]
            node2 = rrt_tree.vertices[edge[1]]
            node1 = (node1[0] + 6 * maze_scaling, node1[1] + 2 * maze_scaling)
            node2 = (node2[0] + 6 * maze_scaling, node2[1] + 2 * maze_scaling)
            lines.append((node1, node2))
        lc = mc.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)
        
        if len(opt_path) > 0:
            path = []
            for i in range(len(opt_path)-1):
                node1 = rrt_tree.vertices[opt_path[i]]
                node2 = rrt_tree.vertices[opt_path[i + 1]]
                node1 = (node1[0] + 6 * maze_scaling, node1[1] + 2 * maze_scaling)
                node2 = (node2[0] + 6 * maze_scaling, node2[1] + 2 * maze_scaling)
                path.append((node1, node2))
            lc2 = mc.LineCollection(path, colors='red', linewidths=1)
            ax.add_collection(lc2)

        if len(states) > 0:
            trace_x = []
            trace_y = []
            for i in range(len(states)):
                trace_x.append(states[i][0] + 6 * maze_scaling)
                trace_y.append(states[i][1] + 2 * maze_scaling)
            ax.plot(trace_x, trace_y, color="k", linewidth=1)

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        plt.savefig("./code/gcsl_ant/fig/rrt_star_tree" + str(timestep) + "_"+str(count) + ".pdf")

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


class AntOmegaMazeEnv(GoalEnv, Serializable):
    def __init__(self, silent=True):
        cls = AntMazeEnv

        maze_id = 'AntOmega'
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

        self.maze_structure = self.maze_env.MAZE_STRUCTURE
        # structure = [
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0,'r',0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],(x)
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                       (y)
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

        # obstacles region:
        # when -2.5<= x <=-0.5 or 5.5<= x <=7.5 , -10.5<= y <=-4.5
        #      -0.5<= x <=5.5, -4.5<= y <=-2.5
        #      1.5<= x <=3.5, -2.5<= y <=1.5
        self.ant_radius = 0.75
        self.wall_collision_buffer = 0.25
        self.obstacles = []
        box_vertical_left = Box_obstacle([-2.5, -2.5], [-0.5, -2.5], [-2.5, -10.5], [-0.5, -10.5], self.ant_radius + self.wall_collision_buffer)
        box_vertical_right = Box_obstacle([5.5, -2.5], [7.5, -2.5], [5.5, -10.5], [7.5, -10.5], self.ant_radius + self.wall_collision_buffer)
        box_vertical_bottom = Box_obstacle([1.5, 1.5], [3.5, 1.5], [1.5, -2.5], [3.5, -2.5], self.ant_radius + self.wall_collision_buffer)
        box_horizontal = Box_obstacle([-0.5, -2.5], [5.5, -2.5], [-0.5, -4.5], [5.5, -4.5], self.ant_radius + self.wall_collision_buffer)
        self.obstacles.append(box_vertical_left)
        self.obstacles.append(box_vertical_right)
        self.obstacles.append(box_vertical_bottom)
        self.obstacles.append(box_horizontal)

        left_border = Box_obstacle([-6.5, 2.5], [-5.5, 2.5], [-6.5, -15.5], [-5.5, -15.5], self.ant_radius + self.wall_collision_buffer)
        right_border = Box_obstacle([10.5, 2.5], [11.5, 2.5], [10.5, -15.5], [11.5, -15.5], self.ant_radius + self.wall_collision_buffer)
        bottom_border = Box_obstacle([-6.5, 2.5], [11.5, 2.5], [-6.5, 1.5], [11.5, 1.5], self.ant_radius + self.wall_collision_buffer)
        upper_border = Box_obstacle([-6.5, -14.5], [11.5, -14.5], [-6.5, -15.5], [11.5, -15.5], self.ant_radius + self.wall_collision_buffer)
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
        return spaces.Box(low=np.array([-5, -14]), high=np.array([10, 1]), dtype=np.float32)

    # actaully, the data it returns is tof the same shape of state, but just contain the final goal information
    def sample_goal(self):
        shape = self.observation_space.shape
        goal = np.zeros(shape)

        # do sample until goal is a feasible one
        goal_candidate = self.goal_space.sample()
        goal_maze_idx = [int((goal_candidate[1] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING),
                         int((goal_candidate[0] + 0.5 * self.maze_env.MAZE_SIZE_SCALING)//self.maze_env.MAZE_SIZE_SCALING)]
        while (self.maze_structure[goal_maze_idx[0] + 15][goal_maze_idx[1] + 6] == 1
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
            if (line.p0[0] > -2.5 and line.p0[0] < -0.5) or (line.p0[0] > 5.5 and line.p0[0] < 7.5):
                t1 = (-10.5 - line.p0[1])/line.dirn[1]
                t2 = (-2.5 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            elif (line.p0[0] > -0.5 and line.p0[0] < 1.5) or (line.p0[0] > 3.5 and line.p0[0] < 5.5):
                t1 = (-4.5 - line.p0[1])/line.dirn[1]
                t2 = (-2.5 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            elif line.p0[0] > 1.5 and line.p0[0] < 3.5:
                t1 = (-2.5 - line.p0[1])/line.dirn[1]
                t2 = (1.5 - line.p0[1])/line.dirn[1]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            else:
                return False
        elif line.dirn[1] == 0:  # y = const
            if line.p0[1] > -10.5 and line.p0[1] < -4.5:
                out_of_left_box = False
                out_of_right_box = False

                t1 = (-2.5 - line.p0[0])/line.dirn[0]
                t2 = (-0.5 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    out_of_left_box = True
                else:
                    out_of_left_box = False

                t3 = (5.5 - line.p0[0])/line.dirn[0]
                t4 = (7.5 - line.p0[0])/line.dirn[0]
                if (t3 < 0 and t4 < 0) or (t3 > line.dist and t4 > line.dist):
                    out_of_right_box = True
                else:
                    out_of_right_box = False

                if out_of_left_box and out_of_right_box:
                    return False
                else:
                    return True
            elif line.p0[1] > -4.5 and line.p0[1] < -2.5:
                t1 = (-2.5 - line.p0[0])/line.dirn[0]
                t2 = (7.5 - line.p0[0])/line.dirn[0]
                if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
                    return False
                else:
                    return True
            elif line.p0[1] > -2.5 and line.p0[1] < 1.5:
                t1 = (1.5 - line.p0[0])/line.dirn[0]
                t2 = (3.5 - line.p0[0])/line.dirn[0]
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
        return state[..., : 2]

    ###
    def _extract_sgoal(self, state):
        return state[..., : 2]

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
                x.append(trajectories[i][j][0] + 6 * self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 15 * self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0: 2], True)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace"+str(i)+".pdf")
            else:
                self.plot_trajectory_fig(x, y, desired_goal_states[i][0: 2], False)
                plt.savefig("./code/gcsl_ant/fig/trace"+str(i)+".pdf")

        return OrderedDict([])

    def plot_trajectory_fig(self, x, y, desired_goal, reached=False):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        plt.clf()
        fig, ax = plt.subplots()
        # Draw goal
        ax.scatter(desired_goal[0] + 6 * maze_scaling, desired_goal[1] + 15 * maze_scaling, s=400, marker='*')

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
            ax.scatter(x[i] + 6 * maze_scaling, y[i] + 15 * maze_scaling, s=10, marker='*')

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, self.maze_env.MAZE_SIZE_SCALING)

        plt.savefig("./code/gcsl_ant/fig/" + self.maze_env._maze_id + "sample_goal_scatter_"+str(self.maze_env._maze_id)+".pdf")

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, rrt_tree, opt_path=[], states=[], count=-1):
        maze_scaling = self.maze_env.MAZE_SIZE_SCALING
        px = [x + 6 * maze_scaling for x, y in rrt_tree.vertices]
        py = [y + 15 * maze_scaling for x, y in rrt_tree.vertices]
        fig, ax = plt.subplots()
        ax.scatter(px, py, c='cyan')
        ax.scatter([x + 6 * maze_scaling for x in goal_pos_x[-goal_scatter_num:]], [y + 15 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*')

        lines = []
        for edge in rrt_tree.edges:
            node1 = rrt_tree.vertices[edge[0]]
            node2 = rrt_tree.vertices[edge[1]]
            node1 = (node1[0] + 6 * maze_scaling, node1[1] + 15 * maze_scaling)
            node2 = (node2[0] + 6 * maze_scaling, node2[1] + 15 * maze_scaling)
            lines.append((node1, node2))
        lc = mc.LineCollection(lines, colors='green', linewidths=2)
        ax.add_collection(lc)
        
        if len(opt_path) > 0:
            path = []
            for i in range(len(opt_path)-1):
                node1 = rrt_tree.vertices[opt_path[i]]
                node2 = rrt_tree.vertices[opt_path[i + 1]]
                node1 = (node1[0] + 6 * maze_scaling, node1[1] + 15 * maze_scaling)
                node2 = (node2[0] + 6 * maze_scaling, node2[1] + 15 * maze_scaling)
                path.append((node1, node2))
            lc2 = mc.LineCollection(path, colors='red', linewidths=1)
            ax.add_collection(lc2)

        if len(states) > 0:
            trace_x = []
            trace_y = []
            for i in range(len(states)):
                trace_x.append(states[i][0] + 6 * maze_scaling)
                trace_y.append(states[i][1] + 15 * maze_scaling)
            ax.plot(trace_x, trace_y, color="k", linewidth=1)

        # Draw maze
        ax = Draw_GridWorld(ax, self.maze_structure, maze_scaling)

        plt.savefig("./code/gcsl_ant/fig/rrt_star_tree" + str(timestep) + "_"+str(count) + ".pdf")

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
