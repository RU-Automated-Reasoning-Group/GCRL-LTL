import argparse
import random

import torch
import gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

from envs import ZonesEnv, ZoneRandomGoalEnv, ZoneRandomGoalContinualEnv
from envs.utils import get_zone_vector
from algo import path_finding, gc_reaching
from rl import GCVNetwork
from sampler import TaskSampler


class TaskConfig:
    def __init__(self, args):
        self.task = args.task
        self.seed = args.seed
        self.policy_path = args.policy_path
        self.gcvf_path = args.gcvf_path
        self.eval_repeats = args.eval_repeats
        self.device = torch.device(args.device)
        self.init_task_params()

    def init_task_params(self):
    
        self.temp = 1.5
        self.value_threshold = 0.85
        self.goals_representation = get_zone_vector()
    
        if self.task == 'avoid':
            self.max_timesteps = 1000
            
        elif self.task == 'chain':
            self.max_timesteps = 2000

        elif self.task == 'traverse':
            self.max_timesteps = 1500

        elif self.task == 'stable':
            self.max_timesteps = 1500


def exp(config):

    task = config.task
    seed = config.seed
    device = config.device

    policy = ActorCriticPolicy.load(path=config.policy_path, device=device)
    gcvf = GCVNetwork.load(path=config.gcvf_path, device=device)

    num_success, num_dangerous = 0, 0
    total_rewards = 0
    total_omega = 0
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
            if task in ('avoid', 'chain'):
                print('[FORMULA]', formula, '[GOALS]', GOALS, '[AVOID]', AVOID_ZONES)
            elif task in ('stable'):
                STABLE_GOAL = GOALS[0]
                print('[FORMULA]', formula, '[GOALS]', STABLE_GOAL, '[AVOID]', '[]')
            elif task in ('traverse'):
                TRAVERSE_GOAL, TRAVERSE_AVOID = GOALS[:2], AVOID_ZONES[:2]
                print('[FORMULA]', formula, '[GOALS]', TRAVERSE_GOAL, '[AVOID]', TRAVERSE_AVOID)

            env = ZoneRandomGoalEnv(
                env=gym.make('Zones-8-v1', map_seed=seed+i, timeout=10000000),  # NOTE: dummy timeout
                primitives_path='models/primitives',
                goals_representation=config.goals_representation,
                use_primitves=True,
                temperature=config.temp,
                device=config.device,
                max_timesteps=config.max_timesteps,
                debug=False if task in ('stable') else True,
            )
            env.reset()

            task_info = gc_reaching(env, policy, gcvf, GOALS, AVOID_ZONES, value_threshold=config.value_threshold, device=device)
            if task_info['complete']:
                num_success += 1
                total_rewards += pow(0.998, env.executed_timesteps)
            elif task_info['dangerous']:
                num_dangerous += 1
            
            if task == 'stable':
                history = task_info['zone_history']
                idx = history.index(STABLE_GOAL)
                num_stable = history.count(STABLE_GOAL) / (1 - idx / config.max_timesteps)
                total_omega += num_stable
            elif task == 'traverse' and not task_info['dangerous']:
                history = task_info['task_history']
                num_zones = len(history)
                total_omega += num_zones / len(TRAVERSE_GOAL)
        
        print('+'*80)
        if task == 'avoid':
            print('[EXP][num_success][{}][num_dangerous][{}]'.format(num_success, num_dangerous))
            print('[Discounted reward][{}]'.format(total_rewards/config.eval_repeats))
        elif task == 'chain':
            print('[EXP][num_success][{}][num_dangerous][{}]'.format(num_success, num_dangerous))
        elif task == 'stable':
            print('[EXP][omega]][{}]'.format(total_omega/config.eval_repeats))
        elif task == 'traverse':
            print('[EXP][omega]][{}]'.format(total_omega/config.eval_repeats))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--task', type=str, default='avoid', choices=('avoid', 'chain', 'stable', 'traverse'))
    parser.add_argument('--policy_path', type=str, default='models/gc_ppo_policy.zip')
    parser.add_argument('--gcvf_path', type=str, default='models/gc_ppo_gcvf.zip')
    parser.add_argument('--eval_repeats', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    task_config = TaskConfig(args)
    exp(task_config)
