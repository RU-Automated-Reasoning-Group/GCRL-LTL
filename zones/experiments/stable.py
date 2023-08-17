import argparse
import random
import time
import sys
sys.path.insert(0, sys.path[0]+"/../")

import torch
import gym
import numpy as np
from stable_baselines3 import PPO

from envs import ZonesEnv
from ltl_wrappers import RandomGoalLTLNormalEnv
from utils import get_named_goal_vector


def task_function(env, model, goal_zone, avoid_zones, value_threshold=0.85, device=torch.device('cuda:0')):

    env.fix_goal(goal_zone)
    ob = env.current_observation()

    num_success = 0
    eventual_reward = 1

    with torch.no_grad():
        while env.executed_timesteps < 1500:  # complete the current subtask
            
            avoid_zones_ob = [env.custom_observation(goal=avoid_zone) for avoid_zone in avoid_zones]
            avoid_zones_vs = []
            for idx in range(len(avoid_zones)):
                x_actor, x_critic = model.policy.mlp_extractor(torch.as_tensor(avoid_zones_ob[idx]).float().to(device))
                avoid_zones_vs.append(model.policy.value_net(x_critic).detach().cpu())
            avoid_zones_vs = np.array(avoid_zones_vs)

            dangerous_zone_indices = np.argwhere(avoid_zones_vs > value_threshold)
            if dangerous_zone_indices.size > 0:

                # NOTE: simple strategy, only avoid the most dangerous action
                # safe_ation + goal_reaching_action (blocked) when V(avoid) > V(goal)
                x_actor, x_critic = model.policy.mlp_extractor(torch.as_tensor(ob).float().to(device))
                goal_v = model.policy.value_net(x_critic)
                most_dangerous_zone_index = np.argmax(avoid_zones_vs).item()

                if avoid_zones_vs[most_dangerous_zone_index] > goal_v:

                    # safe_action
                    action_distribution = model.policy.get_distribution(torch.from_numpy(avoid_zones_ob[most_dangerous_zone_index]).unsqueeze(dim=0).to(device))
                    action_probs = action_distribution.distribution.probs
                    safe_action = torch.argmin(action_probs, dim=1)

                    # (blocked) goal_reaching_aciton
                    avoid_zone_action, _states = model.predict(avoid_zones_ob[most_dangerous_zone_index], deterministic=True)
                    action_distribution = model.policy.get_distribution(torch.from_numpy(ob).unsqueeze(dim=0).to(device))
                    action_probs = action_distribution.distribution.probs
                    dangerous_mask = torch.ones(4).to(device)
                    dangerous_mask[avoid_zone_action] = 0
                    goal_reaching_action = torch.argmax(action_probs * dangerous_mask, dim=1)

                    ob, reward, eval_done, info = env.step({
                        'action': [safe_action, goal_reaching_action],
                        'distribution': [1, 1],
                    })
                else:
                    action, _states = model.predict(ob, deterministic=True)
                    ob, reward, eval_done, info = env.step(action)
                
            else:
                action, _states = model.predict(ob, deterministic=True)
                ob, reward, eval_done, info = env.step(action)
            
            if info['zone'] == goal_zone:
                num_success += 1
                eventual_reward *= 0.998
            else:
                eventual_reward *= 0.998

    return num_success, eventual_reward


def translate(GOALS, _AVOID_ZONES):

    GOALS = [z.capitalize() for z in GOALS]
    AVOID_ZONES = []
    for Z in _AVOID_ZONES:
        AVOID_ZONES.append([z.capitalize() for z in Z])
    
    return GOALS, AVOID_ZONES


def experiment(args):

    seed = args.seed
    temp = args.temp
    rl_model_path = args.rl_model_path
    eval_repeats = args.eval_repeats
    value_threshold = args.value_threshold
    device = torch.device(args.device)
    max_timesteps = args.max_timesteps
    letter = args.letter

    model = PPO.load(rl_model_path, device=device)
    goals_representation = get_named_goal_vector()
    total_num_success = 0
    total_eventual_rewards = 0
    GOALS = ['J', 'W', 'R', 'Y']

    with torch.no_grad():
        
        for i in range(eval_repeats):

            # NOTE: STABILIZE
            GOAL = letter if letter else GOALS[i % 4]
            AVOID = []

            env = RandomGoalLTLNormalEnv(
                env=gym.make('Zones-8-v1', map_seed=seed+int(time.time() * 1000000) % 100, timeout=10000000000),  # NOTE: dummy timeout
                primitives_path='models/primitives',
                goals_representation=goals_representation,
                use_primitves=True,
                temperature=temp,
                device=device,
                max_timesteps=max_timesteps,
                debug=False,
            )
            env.reset()

            num_success, eventual_rewards = task_function(env, model, GOAL, AVOID, value_threshold=value_threshold, device=device)
            total_num_success += num_success
            total_eventual_rewards += eventual_rewards
            print('[num_success]', num_success, '[eventual_rewards]', eventual_rewards)

    return total_num_success / eval_repeats, total_eventual_rewards / eval_repeats


if __name__ == '__main__':

    rl_model_path = 'models/goal-conditioned/best_model_ppo_8'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--temp', type=float, default=1.5)
    parser.add_argument('--max_timesteps', type=int, default=1000000000),
    parser.add_argument('--rl_model_path', type=str, default=rl_model_path)
    parser.add_argument('--eval_repeats', type=int, default=20)
    parser.add_argument('--value_threshold', type=float, default=0.85)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--letter', type=str, default='', choices=('J', 'W', 'R', 'Y'))
    
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_success, reward = experiment(args)
    print('{}|{}'.format(num_success, reward))
