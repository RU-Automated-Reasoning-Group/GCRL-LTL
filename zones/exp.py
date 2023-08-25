import argparse
import random

import torch
import gym
import numpy as np
from stable_baselines3 import PPO

from envs import ZonesEnv, ZoneRandomGoalEnv
from envs.utils import get_named_goal_vector
from algo import path_finding, reaching
from sampler import TaskSampler


class TaskConfig:
    def __init__(self, args):
        self.task = args.task
        self.seed = args.seed
        self.rl_model_path = args.rl_model_path
        self.eval_repeats = args.eval_repeats
        self.device = torch.device(args.device)
        self.init_task_params()

    def init_task_params(self):
    
        self.temp = 1.5
        self.value_threshold = 0.85
        self.goals_representation = get_named_goal_vector()
    
        if self.task == 'avoid':
            self.max_timesteps = 1000
            
        elif self.task == 'chain':
            self.max_timesteps = 1500

        elif self.task == 'traverse':
            self.max_timesteps = 1500

        elif self.task == 'stable':
            self.max_timesteps = 1500


def exp(config):

    task = config.task
    seed = config.seed
    device = config.device

    model = PPO.load(config.rl_model_path, device=device)
    
    num_success, num_dangerous = 0, 0
    total_rewards = 0
    total_task_history = []
    aps = ['J', 'W', 'R', 'Y']
    sampler = TaskSampler(task=task, aps=aps)

    with torch.no_grad():

        print('-' * 30)
        for i in range(config.eval_repeats):

            random.seed(seed + i)

            formula = sampler.sample()
            GOALS, AVOID_ZONES = path_finding(formula)
            
            print('+'*80)
            print('[ITERATION][{}]'.format(i))
            print('[FORMULA]', formula, '[GOALS]', GOALS, '[AVOID]', AVOID_ZONES)

            env = ZoneRandomGoalEnv(
                env=gym.make('Zones-8-v1', map_seed=seed+i, timeout=10000000),  # NOTE: dummy timeout
                primitives_path='models/primitives',
                goals_representation=config.goals_representation,
                use_primitves=True,
                temperature=config.temp,
                device=config.device,
                max_timesteps=config.max_timesteps,
                debug=True,
            )
            env.reset()

            task_info = reaching(env, model, GOALS, AVOID_ZONES, value_threshold=config.value_threshold, device=device)
            if task_info['complete']:
                num_success += 1
                total_rewards += pow(0.998, env.executed_timesteps)
            elif task_info['dangerous']:
                num_dangerous += 1
            total_task_history += task_info['task_history']
            print('+'*80)
                
        print('[EXP][success][{}][num_dangerous][{}]'.format(num_success, num_dangerous))
        if task == 'avoid':
            print('[Discounted reward][{}]'.format(total_rewards/config.eval_repeats))
        elif task == 'chain':
            pass
        elif task == 'stable':
            pass
        elif task == 'traverse':
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--task', type=str, default='avoid', choices=('avoid', 'chain'))
    parser.add_argument('--rl_model_path', type=str, default='models/goal-conditioned/best_model_ppo_8')
    parser.add_argument('--eval_repeats', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    task_config = TaskConfig(args)
    exp(task_config)
