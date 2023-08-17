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
# import gymnasium as gymn


class FetchSlideEnv(GoalEnv, Serializable):
    def __init__(self, max_episode_steps=100):
        self.wrapped_env = gym.make(
            "FetchSlide-v2",max_episode_steps )  # wrapped_env
        self.wrapped_env.reset()
        self.quick_init(locals())

    @property
    def observation_space(self):
        return self.wrapped_env.observation_space['observation']

    @property
    # action space is a 4-dim vector which indicate four primitives
    def action_space(self):
        return self.wrapped_env.action_space

    @property
    def state_space(self):
        return self.wrapped_env.observation_space['observation']

    @property
    def goal_space(self):
        return self.wrapped_env.observation_space['achieved_goal']

    # must reset the nev when sampling a goal position
    def sample_goal(self):
        obs = self.wrapped_env.reset()
        return obs['desired_goal']

    def reset(self):
        """
        Resets the environment and returns a state vector
        Returns: The initial state
        """
        obs = self.wrapped_env.reset()
        return obs

    # action 4-dim
    def step(self, action):
        """
        Runs 1 step of simulation

        Returns:
            A tuple containing:
                next_state
                reward (always 0)
                done
                infos
        """

        obs, reward, terminated, truncated, info = self.maze_env.step(action)

        done = True if info['is_success']==1 else False

        return obs, reward, done, info

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
                x.append(trajectories[i][j][0] + 4 *
                         self.maze_env.MAZE_SIZE_SCALING)
                y.append(trajectories[i][j][1] + 4 *
                         self.maze_env.MAZE_SIZE_SCALING)
            if len(success_vec) != 0 and success_vec[i] == 1:
                self.plot_trajectory_fig(
                    x, y, desired_goal_states[i][0: 2], True)
                plt.savefig("./code/gcsl_ant/fig/(succeed)trace"+str(i)+".pdf")
            else:
                self.plot_trajectory_fig(
                    x, y, desired_goal_states[i][0: 2], False)
                plt.savefig("./code/gcsl_ant/fig/trace"+str(i)+".pdf")

        return OrderedDict([])

    def plot_trajectory_fig(self, x, y, desired_goal, reached=False):
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
                   [y + 4 * maze_scaling for y in goal_pos_y[-goal_scatter_num:]], s=100, marker='*')

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
