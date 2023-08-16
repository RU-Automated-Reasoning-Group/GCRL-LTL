"""
This is a simple wrapper that will include LTL goals to any given environment.
It also progress the formulas as the agent interacts with the envirionment.

However, each environment must implement the followng functions:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.
    - *get_propositions(...)*: Maps the objects in the environment to a set of
                            propositions that can be referred to in LTL.

Notes about LTLEnv:
    - The episode ends if the LTL goal is progressed to True or False.
    - If the LTL goal becomes True, then an extra +1 reward is given to the agent.
    - If the LTL goal becomes False, then an extra -1 reward is given to the agent.
    - Otherwise, the agent gets the same reward given by the original environment.
"""

import os
import torch
import numpy as np
import gym
from gym import spaces
import ltl_progression
from ltl_samplers import getLTLSampler, SequenceSampler
from stable_baselines3 import PPO


class LTLEnv(gym.Wrapper):
    def __init__(self, env, progression_mode="full", ltl_sampler=None, intrinsic=0.0):
        """
        LTL environment
        --------------------
        It adds an LTL objective to the current environment
            - The observations become a dictionary with an added "text" field
              specifying the LTL objective
            - It also automatically progress the formula and generates an
              appropriate reward function
            - However, it does requires the user to define a labeling function
              and a set of training formulas
        progression_mode:
            - "full": the agent gets the full, progressed LTL formula as part of the observation
            - "partial": the agent sees which propositions (individually) will progress or falsify the formula
            - "none": the agent gets the full, original LTL formula as part of the observation
        """
        super().__init__(env)
        self.progression_mode   = progression_mode
        self.propositions = self.env.get_propositions()
        self.sampler = getLTLSampler(ltl_sampler, self.propositions)

        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.known_progressions = {}
        self.intrinsic = intrinsic


    def sample_ltl_goal(self):
        # This function must return an LTL formula for the task
        # Format:
        #(
        #    'and',
        #    ('until','True', ('and', 'd', ('until','True',('not','c')))),
        #    ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
        #)
        # NOTE: The propositions must be represented by a char
        raise NotImplementedError

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        raise NotImplementedError

    def reset(self):
        self.known_progressions = {}
        self.obs = self.env.reset()

        # Defining an LTL goal
        self.ltl_goal     = self.sample_ltl_goal()
        self.ltl_original = self.ltl_goal

        # Adding the ltl goal to the observation
        if self.progression_mode == "partial":
            ltl_obs = {'features': self.obs,'progress_info': self.progress_info(self.ltl_goal)}
        else:
            ltl_obs = {'features': self.obs,'text': self.ltl_goal}
        return ltl_obs


    def step(self, action):
        int_reward = 0
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # progressing the ltl formula
        truth_assignment = self.get_events(self.obs, action, next_obs)
        self.ltl_goal = self.progression(self.ltl_goal, truth_assignment)
        self.obs      = next_obs

        # Computing the LTL reward and done signal
        ltl_reward = 0.0
        ltl_done   = False
        if self.ltl_goal == 'True':
            ltl_reward = 1.0
            ltl_done   = True
        elif self.ltl_goal == 'False':
            ltl_reward = -1.0
            ltl_done   = True
        else:
            ltl_reward = int_reward

        # Computing the new observation and returning the outcome of this action
        if self.progression_mode == "full":
            ltl_obs = {'features': self.obs,'text': self.ltl_goal}
        elif self.progression_mode == "none":
            ltl_obs = {'features': self.obs,'text': self.ltl_original}
        elif self.progression_mode == "partial":
            ltl_obs = {'features': self.obs, 'progress_info': self.progress_info(self.ltl_goal)}
        else:
            raise NotImplementedError

        reward  = original_reward + ltl_reward
        done    = env_done or ltl_done
        return ltl_obs, reward, done, info

    def progression(self, ltl_formula, truth_assignment):

        if (ltl_formula, truth_assignment) not in self.known_progressions:
            result_ltl = ltl_progression.progress_and_clean(ltl_formula, truth_assignment)
            self.known_progressions[(ltl_formula, truth_assignment)] = result_ltl

        return self.known_progressions[(ltl_formula, truth_assignment)]


    # # X is a vector where index i is 1 if prop i progresses the formula, -1 if it falsifies it, 0 otherwise.
    def progress_info(self, ltl_formula):
        propositions = self.env.get_propositions()
        X = np.zeros(len(self.propositions))

        for i in range(len(propositions)):
            progress_i = self.progression(ltl_formula, propositions[i])
            if progress_i == 'False':
                X[i] = -1.
            elif progress_i != ltl_formula:
                X[i] = 1.
        return X

    def sample_ltl_goal(self):
        # NOTE: The propositions must be represented by a char
        # This function must return an LTL formula for the task
        formula = self.sampler.sample()

        if isinstance(self.sampler, SequenceSampler):
            def flatten(bla):
                output = []
                for item in bla:
                    output += flatten(item) if isinstance(item, tuple) else [item]
                return output

            length = flatten(formula).count("and") + 1
            self.env.timeout = 25 # 10 * length

        return formula


    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()


class NoLTLWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Removes the LTL formula from an LTLEnv
        It is useful to check the performance of off-the-shelf agents
        """
        super().__init__(env)
        self.observation_space = env.observation_space
        # self.observation_space =  env.observation_space['features']

    def reset(self):
        obs = self.env.reset()
        # obs = obs['features']
        # obs = {'features': obs}
        return obs

    def step(self, action):
        # executing the action in the environment
        obs, reward, done, info = self.env.step(action)
        # obs = obs['features']
        # obs = {'features': obs}
        return obs, reward, done, info

    def get_propositions(self):
        return list([])


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


class RandomGoalLTLNormalEnv(gym.Wrapper):

    PRIMITVE_OBS_DIM = 12
    DT = 0.002

    def __init__(self, env, primitives_path, goals_representation, temperature=1.25, use_primitves=True, rewards=[0, 1], device=torch.device('cuda:0'), max_timesteps=1000):
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
        self.goal_is_fixed = False
        self.others_reward, self.success_reward = rewards[0], rewards[1]
        self.last_robot_pos = None
        self.temperature = temperature
        self.max_timesteps = max_timesteps
        self.executed_timesteps = 0

    def is_alive(self):
        return self.executed_timesteps < self.max_timesteps

    def current_observation(self):
        obs = self.env.obs()
        goal = self.goals_representation[self.goals[self.goal_index]][0]
        return np.concatenate((obs, goal))

    def custom_observation(self, goal: str):
        obs = self.env.obs()
        goal = self.goals_representation[goal][0]
        return np.concatenate((obs, goal))

    def fix_goal(self, goal):
        assert goal in self.goals
        self.goal_index = self.goals.index(goal)
        self.goal_is_fixed = True

    def reset(self):
        self.executed_timesteps = 0
        #self.goal_index = int(time.time() * 10000) % 4
        if not self.goal_is_fixed:
            self.goal_index = (self.goal_index + 1) % 4
        obs = super().reset()
        self.last_robot_pos = self.robot_pos
        goal = self.goals_representation[self.goals[self.goal_index]][0]
        return np.concatenate((obs, goal))

    def current_goal(self):
        return self.goals[self.goal_index]

    def _translate_primitive(self, action):
        
        #ob = self.current_observation()
        ob = self.env.obs()  # NOTE: more efficient
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

    def step(self, action, skip_translate=False):
        if self.use_primitves and not skip_translate:
            action = self.translate_primitive(action)
        # NOTE: move faster, life is easier
        next_obs, original_reward, env_done, info = self.env.step(action * self.temperature)
        xy_velocity = (self.last_robot_pos[:2].copy() - self.env.robot_pos[:2].copy()) / self.DT
        self.last_robot_pos = self.env.robot_pos
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
                #print('Yes I reached the {} goal'.format(self.current_goal()))
                info = {'zone': self.current_goal(), 'task': self.current_goal()}
            else:
                reward = self.others_reward
                done = env_done
                info = {'zone': truth_assignment, 'task': self.current_goal()}

        return self.current_observation(), reward, done, info

