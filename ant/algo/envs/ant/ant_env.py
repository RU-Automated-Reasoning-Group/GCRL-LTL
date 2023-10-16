import os
import gym
import numpy as np

from algo.envs.ant.ant_maze_env import AntMazeEnv


class AntFall(gym.Env):
    
    def __init__(self):
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

        self.maze_env = cls(**gym_mujoco_kwargs) # wrapped_env
        self.maze_env.reset()

    @property
    def observation_space(self):
        ant_obs = self.maze_env._get_obs()
        box_obs = self.maze_env.wrapped_env.data.get_body_xpos('movable_2_2')
        shape = np.concatenate([ant_obs, box_obs]).shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        return self.maze_env.action_space

    def reset(self):
        ant_obs = self.maze_env.reset()
        box_obs = self.maze_env.wrapped_env.data.get_body_xpos('movable_2_2')
        
        obs = np.concatenate([ant_obs, box_obs])
        return obs
    
    def step(self, action):
        ob, reward, done, info = self.maze_env.step(action)

        box_ob = self.maze_env.wrapped_env.data.get_body_xpos('movable_2_2')
        ob = np.concatenate([ob, box_ob])

        distance = np.linalg.norm(ob[:2] - np.array([0, 27]))
        reward = -distance
        done = True if distance < 1.0 else False
        progress = (1 - (distance - 1) / (27 - 1)) * 100

        info['finished'] = False
        info['distance'] = (distance - 1) / (27 - 1) if distance > 1.0 else 0.0
        info['progress'] = progress

        return ob, reward, done, info


class AntPush(gym.Env):

    def __init__(self):
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

    @property
    def observation_space(self):
        shape = self.maze_env._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        return self.maze_env.action_space

    def reset(self):
        return self.maze_env.reset()

    def step(self, action):
        ob, reward, done, info = self.maze_env.step(action)

        distance = np.linalg.norm(ob[:2] - np.array([0, 18]))
        reward = -distance
        done = True if distance < 1.0 else False
        progress = (1 - (distance - 1) / (18 - 1)) * 100

        info['finished'] = False
        info['distance'] = (distance - 1) / (18 - 1) if distance > 1.0 else 0.0
        info['progress'] = progress

        return ob, reward, done, info


class AntMaze(gym.Env):

    def __init__(self):
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

    @property
    def observation_space(self):
        shape = self.maze_env._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        return self.maze_env.action_space

    def reset(self):
        return self.maze_env.reset()

    def step(self, action):
        ob, reward, done, info = self.maze_env.step(action)

        distance = np.linalg.norm(ob[:2] - np.array([0, 16]))
        reward = -distance
        done = True if distance < 1.0 else False
        progress = (1 - (distance - 1) / (16 - 1)) * 100

        info['finished'] = False
        info['distance'] = (distance - 1) / (16 - 1) if distance > 1.0 else 0.0
        info['progress'] = progress

        return ob, reward, done, info
