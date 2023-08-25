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


def main(args):

    seed = args.seed
    temp = args.temp
    rl_model_path = args.rl_model_path
    eval_repeats = args.eval_repeats
    value_threshold = args.value_threshold
    device = torch.device(args.device)
    max_timesteps = args.max_timesteps

    model = PPO.load(rl_model_path, device=device)
    num_success = 0
    num_J_success = 0
    num_W_success = 0
    num_R_success = 0
    num_Y_success = 0
    num_dangerous = 0

    total_rewards = 0
    goals_representation = get_named_goal_vector()
    sampler = TaskSampler(task='avoid', aps=['J', 'W', 'R', 'Y'])

    with torch.no_grad():

        for i in range(eval_repeats):

            random.seed(seed + i)

            formula = sampler.sample()
            GOALS, AVOID_ZONES = path_finding(formula)
            
            print('+'*80)
            print('[ITERATION][{}]'.format(i))
            print('[FORMULA]', formula, '[GOALS]', GOALS, '[AVOID]', AVOID_ZONES)

            stage_index = 0
            env = ZoneRandomGoalEnv(
                env=gym.make('Zones-8-v1', map_seed=seed+i, timeout=10000000),  # NOTE: dummy timeout
                primitives_path='models/primitives',
                goals_representation=goals_representation, 
                use_primitves=True,
                temperature=temp,
                device=device,
                max_timesteps=max_timesteps,
                debug=True,
            )
            env.reset()

            task_info = reaching(env, model, GOALS, AVOID_ZONES, value_threshold=value_threshold, seed=seed+i, device=device)
                complete, history = task_info['complete', 'history']
                if task_info['complete']:
                    if task_info['reach_zone'] == 'J':
                        num_J_success += 1
                    elif task_info['reach_zone'] == 'W':
                        num_W_success += 1
                    elif subtask_info['reach_zone'] == 'R':
                        num_R_success += 1
                    elif subtask_info['reach_zone'] == 'Y':
                        num_Y_success += 1
                    if stage_index == len(GOALS) - 1:
                        num_success += 1
                        total_rewards += pow(0.998, env.executed_timesteps)
                else:
                    if subtask_info['dangerous']:
                        num_dangerous += 1
                    break

            print('[EVAL][total success][{}][J success][{}][W success][{}][R success][{}][Y success][{}]'.format(num_success, num_J_success, num_W_success, num_R_success, num_Y_success))
            print('[num_dangerous]{}'.format(num_dangerous))
            print('[Discounted reward][{}]'.format(total_rewards/(i+1)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temp', type=float, default=1.5)
    parser.add_argument('--max_timesteps', type=int, default=1000),
    parser.add_argument('--rl_model_path', type=str, default='models/goal-conditioned/best_model_ppo_8')
    parser.add_argument('--eval_repeats', type=int, default=10)
    parser.add_argument('--value_threshold', type=float, default=0.85)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(args)
