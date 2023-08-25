import argparse
import random
import time
import sys

import torch
import gym
import numpy as np
from stable_baselines3 import PPO

from envs import ZonesEnv, ZoneRandomGoalEnv
from envs.utils import get_named_goal_vector
from algo import path_finding
from sampler import TaskSampler


def task_function(env, model, goals, avoid_zones, value_threshold=0.85, device=torch.device('cpu')):

    goal_zones = goals
    zone_index = 0
    env.fix_goal(goal_zones[zone_index])
    ob = env.current_observation()

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

            if info['zone'] == goal_zones[zone_index]:
                zone_index += 1
                if zone_index == len(goal_zones):
                    break
                env.fix_goal(goal_zones[zone_index])
    
    rounds = zone_index / 4
    return rounds


def experiment(args):

    seed = args.seed
    temp = args.temp
    rl_model_path = args.rl_model_path
    eval_repeats = args.eval_repeats
    value_threshold = args.value_threshold
    device = torch.device(args.device)
    max_timesteps = args.max_timesteps

    model = PPO.load(rl_model_path, device=device)
    
    total_rounds = 0
    goals_representation = get_named_goal_vector()
    sampler = TaskSampler(task='chain', aps=['J', 'W', 'R', 'Y'])

    with torch.no_grad():

        print('-' * 30)
        for i in range(eval_repeats):

            formula = sampler.sample()
            GOALS, AVOID_ZONES = path_finding(formula)
            
            env = ZoneRandomGoalEnv(
                env=gym.make('Zones-8-v1', map_seed=seed+int(time.time() * 1000000) % 100, timeout=10000000000),  # NOTE: dummy timeout
                primitives_path='models/primitives',
                goals_representation=goals_representation, 
                use_primitves=True,
                temperature=temp,
                device=device,
                max_timesteps=max_timesteps,
                debug=True,
            )
            env.reset()

            print('GOALS', GOALS)
            rounds = task_function(env, model, GOALS, AVOID_ZONES, value_threshold=value_threshold, device=device)
            total_rounds += rounds
            print('-' * 30)

    return total_rounds / eval_repeats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--temp', type=float, default=1.5)
    parser.add_argument('--max_timesteps', type=int, default=1000000000),
    parser.add_argument('--rl_model_path', type=str, default='models/goal-conditioned/best_model_ppo_8')
    parser.add_argument('--eval_repeats', type=int, default=20)
    parser.add_argument('--value_threshold', type=float, default=0.85)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    rounds = experiment(args)
    print('{}'.format(rounds))
