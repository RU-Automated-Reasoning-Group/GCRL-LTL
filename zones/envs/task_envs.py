import os
import torch
import numpy as np
import gym
from stable_baselines3 import PPO

# DEBUG
import copy
import random


class ZonePrimitiveEnv(gym.Wrapper):

    OBS_DIM = 12
    DT = 0.002

    def __init__(self, env, direction='pos_x'):
        super().__init__(env)
        self.direction = direction
        self.observation_space = gym.spaces.Box(high=np.inf, low=-np.inf, shape=(self.OBS_DIM,))  # NOTE: no zones information
        self.action_space = self.env.action_space
        self.last_robot_pos = None

        assert self.direction in ['pos_x', 'neg_x', 'pos_y', 'neg_y']

    def reset(self):
        ob = super().reset()
        self.last_robot_pos = self.robot_pos

        return ob
    
    def reward_func(self, xy_velocity):

        vx, vy = xy_velocity[0], xy_velocity[1]

        if self.direction == 'pos_x':
            reward = vx - 0.05 * np.abs(vy)
        elif self.direction == 'neg_x':
            reward = -vx - 0.05 * np.abs(vy)
        elif self.direction == 'pos_y':
            reward = -0.05 * np.abs(vx) + vy
        elif self.direction == 'neg_y':
            reward = -0.05 * np.abs(vx) - vy

        return reward

    def step(self, action):

        ob, original_reward, env_done, info = self.env.step(action)
        xy_velocity = (self.last_robot_pos[:2].copy() - self.robot_pos[:2].copy()) / self.DT
        self.last_robot_pos = self.robot_pos

        return ob[:self.OBS_DIM], self.reward_func(xy_velocity), env_done, info


class ZoneRandomGoalEnv(gym.Wrapper):

    PRIMITVE_OBS_DIM = 12
    DT = 0.002

    def __init__(self, env, primitives_path, goals_representation, temperature=1.25, use_primitves=True, rewards=[0, 1], device=torch.device('cpu'), max_timesteps=1000, debug=False):
        super().__init__(env)
        self.goals = ['J', 'W', 'R', 'Y']
        self.goal_index = 0
        self.goals_representation = goals_representation
        self.propositions = self.env.get_propositions()
        self.use_primitves = use_primitves
        self.primitives_path = primitives_path
        self.primitives = []
        for direction in ['pos_x', 'neg_x', 'pos_y', 'neg_y']:
            self.primitives.append(PPO.load(os.path.join(self.primitives_path, direction), device=device))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)
        if self.use_primitves:
            self.action_space = gym.spaces.Discrete(len(self.primitives))
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.others_reward, self.success_reward = rewards[0], rewards[1]
        self.temperature = temperature
        self.max_timesteps = max_timesteps
        self.executed_timesteps = 0
        self.debug = debug

    def is_alive(self):
        return self.executed_timesteps < self.max_timesteps

    def current_observation(self):
        obs = self.env.obs()
        goal = self.goals_representation[self.goals[self.goal_index]]
        return np.concatenate((obs, goal))

    def custom_observation(self, goal: str):
        obs = self.env.obs()
        goal = self.goals_representation[goal]
        return np.concatenate((obs, goal))

    def reset(self):
        self.executed_timesteps = 0
        self.goal_index = (self.goal_index + 1) % len(self.goals)
        obs = super().reset()
        goal = self.goals_representation[self.goals[self.goal_index]]
        return np.concatenate((obs, goal))

    def current_goal(self):
        return self.goals[self.goal_index]

    def _translate_primitive(self, action):        
        ob = self.env.obs()
        index = action
        primitive_ob = ob[:self.PRIMITVE_OBS_DIM]
        action, _ = self.primitives[index].predict(primitive_ob, deterministic=True)
        
        return action

    def translate_primitive(self, action):
        if isinstance(action, dict):
            act = np.zeros((2,), dtype=np.float32)
            for idx, a in enumerate(action['action']):
               act += self._translate_primitive(a) * action['distribution'][idx]
            return act
        else:
            return self._translate_primitive(action)

    def step(self, action):
        if self.use_primitves:
            action = self.translate_primitive(action)
        # NOTE: move faster, life is easier
        next_obs, original_reward, env_done, info = self.env.step(action * self.temperature)
        truth_assignment = self.env.get_events()
        self.executed_timesteps += 1

        if not truth_assignment:
            reward = self.others_reward
            done = env_done
            info = {'zone': None, 'task': self.current_goal()}
        else:
            if self.current_goal() in truth_assignment:  # TODO: improve this
                reward = self.success_reward
                done = True
                if self.debug:
                    print('Reach zone [{}]'.format(self.current_goal()))
                info = {'zone': self.current_goal(), 'task': self.current_goal()}
            else:
                reward = self.others_reward
                done = env_done
                info = {'zone': truth_assignment, 'task': self.current_goal()}

        return self.current_observation(), reward, done, info


class ZoneRandomGoalContinualEnv(gym.Wrapper):

    PRIMITVE_OBS_DIM = 12
    DT = 0.002

    def __init__(self, env, primitives_path, zones_representation, temperature=1.25, use_primitves=True, rewards=[0, 1], device=torch.device('cpu'), max_timesteps=1000, debug=False):
        super().__init__(env)

        self.starts = ['J', 'W', 'R', 'Y', 'ANYWHERE']
        self.goals = ['J', 'W', 'R', 'Y']
        self.start_index = 0
        self.goal_index = 0
        self.zones_representation = zones_representation
        self.propositions = self.env.get_propositions()
        self.use_primitves = use_primitves
        self.primitives_path = primitives_path
        self.primitives = []
        for direction in ['pos_x', 'neg_x', 'pos_y', 'neg_y']:
            self.primitives.append(PPO.load(os.path.join(self.primitives_path, direction), device=device))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(124,), dtype=np.float32)
        if self.use_primitves:
            self.action_space = gym.spaces.Discrete(len(self.primitives))
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.others_reward, self.success_reward = rewards[0], rewards[1]
        self.temperature = temperature
        self.max_timesteps = max_timesteps
        self.executed_timesteps = 0
        self.debug = debug

        # DEBUG
        self.saved_envs = []

    def is_alive(self):
        return self.executed_timesteps < self.max_timesteps

    def current_observation(self):
        obs = self.env.obs()
        start = self.zones_representation[self.starts[self.start_index]]
        goal = self.zones_representation[self.goals[self.goal_index]]
        return np.concatenate((obs, start, goal))

    def custom_observation(self, start:str, goal: str):
        obs = self.env.obs()
        start = self.zones_representation[start]
        goal = self.zones_representation[goal]
        return np.concatenate((obs, start, goal))

    def reset(self):
        self.executed_timesteps = 0

        # DEBUG
        if len(self.saved_envs) > 100 and random.random() > 0.5:
            self.env, start = copy.deepcopy(random.choice(self.saved_envs))
            while True:
                goal = random.choice(self.goals)
                if goal != start:
                    break
            self.set_start(start)
            self.set_goal(goal)
            obs = self.env.obs()
        else:
            self.start_index = self.starts.index('ANYWHERE')
            self.goal_index = (self.goal_index + 1) % len(self.goals)
            obs = super().reset()

        start_vector = self.zones_representation[self.starts[self.start_index]]
        goal_vector = self.zones_representation[self.goals[self.goal_index]]
        return np.concatenate((obs, start_vector, goal_vector))

    def current_start(self):
        return self.starts[self.start_index]

    def current_goal(self):
        return self.goals[self.goal_index]
    
    def set_start(self, start:str):
        self.start_index = self.starts.index(start)

    def set_goal(self, goal:str):
        self.goal_index = self.goals.index(goal)

    def _translate_primitive(self, action):
        ob = self.env.obs()
        index = action
        primitive_ob = ob[:self.PRIMITVE_OBS_DIM]
        action, _ = self.primitives[index].predict(primitive_ob, deterministic=True)
        
        return action

    def translate_primitive(self, action):
        if isinstance(action, dict):
            act = np.zeros((2,), dtype=np.float32)
            for idx, a in enumerate(action['action']):
               act += self._translate_primitive(a) * action['distribution'][idx]
            return act
        else:
            return self._translate_primitive(action)

    def step(self, action):
        if self.use_primitves:
            action = self.translate_primitive(action)
        # NOTE: move faster, life is easier
        next_obs, original_reward, env_done, info = self.env.step(action * self.temperature)
        truth_assignment = self.env.get_events()
        self.executed_timesteps += 1

        if not truth_assignment:
            reward = self.others_reward
            done = env_done
            info = {'zone': None, 'task': self.current_goal()}
        else:
            if self.current_goal() in truth_assignment:  # TODO: improve this
                reward = self.success_reward
                done = True
                if self.debug:
                    print('Reach zone [{}]'.format(self.current_goal()))
                info = {'zone': self.current_goal(), 'task': self.current_goal()}
            else:
                reward = self.others_reward
                done = env_done
                info = {'zone': truth_assignment, 'task': self.current_goal()}

        # DEBUG: the prob to store should be considered carefully
        if done:
            print('[Prepare to save]')
            print(self.env)
            self.saved_envs.append((copy.deepcopy(self.env), self.current_goal()))
            print('[SAVE]', self.current_goal(), len(self.saved_envs))

        return self.current_observation(), reward, done, info
