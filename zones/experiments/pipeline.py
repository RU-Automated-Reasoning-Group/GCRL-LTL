import argparse
import random
import sys
sys.path.insert(0, sys.path[0]+"/../")

import torch
import gym
import numpy as np
import spot
from stable_baselines3 import PPO

from envs import ZonesEnv
from ltl_wrappers import RandomGoalLTLNormalEnv
from utils import get_named_goal_vector
from scc import SCC_Algorithm
from types import SimpleNamespace
from ltl_cmd import gltl2ba
from ltl_progression import progress, _get_spot_format
from ltl_samplers import getLTLSampler


def get_ltl_args(formula):
    
    args = SimpleNamespace()
    args.formula = formula
    args.file = None

    args.s = False
    args.d = False
    args.l = False
    args.p = False
    args.o = False
    args.c = False
    args.a = False

    args.graph = False
    args.output_graph = open('ba_test', 'w')
    args.dot = False
    args.output_dot = open('ba_test.gv', 'w')

    return args


def parse_ltl_path(ltl_path, translation_function=None):

    if translation_function is None:
        def translate(word):
            return word.upper()
    else:
        translate = translation_function

    GOALS, AVOID_ZONES = [], []
    for f in ltl_path:
        avoid_zones = []
        f = f.replace('(', '').replace(')', '').split('&')
        f = [_f.strip() for _f in f]
        for _f in f:
            if '!' not in _f:
                GOALS.append(translate(_f))
            else:
                avoid_zones.append(translate(_f.replace('!', '')))
        AVOID_ZONES.append(avoid_zones)

    return GOALS, AVOID_ZONES


def reformat_ltl(formula):
    ltl = progress(formula, '')
    ltl_spot = _get_spot_format(ltl)
    f = spot.formula(ltl_spot)
    f = spot.simplify(f)
    f = str(f).replace('&', '&&').replace('"', '').lower()
    f = f.replace('u', 'U').replace('f', '<>').replace('g', '[]').replace('x', 'X')

    return f


def ltl_subtask_v0(env, model, goal_zone, avoid_zones, seed, value_threshold=0.85, device=torch.device('cpu')):

    env.fix_goal(goal_zone)
    ob = env.current_observation()

    with torch.no_grad():
        eval_done = False
        while not eval_done and env.is_alive():  # complete the current subtask
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

            if info['zone'] in avoid_zones:
                print('[Dangerous !][reach {} avoid {}][overlap with {}][seed][{}]'.format(goal_zone, avoid_zones, info['zone'], seed))
                return (ob, reward, eval_done, info), {'complete': False, 'dangerous': True, 'reach_zone': None}
            
            if info['zone'] == goal_zone:
                return (ob, reward, eval_done, info), {'complete': True, 'dangerous': False, 'reach_zone': goal_zone}

    return (ob, 0, True, {}), {'complete': False, 'dangerous': False, 'reach_zone': None}


def ltl_subtask_v1(env, model, goal_zone, avoid_zones, seed, value_threshold=0.85, device=torch.device('cpu')):

    env.fix_goal(goal_zone)
    ob = env.current_observation()

    with torch.no_grad():
        eval_done = False
        while not eval_done and env.is_alive():  # complete the current subtask
            avoid_zones_ob = [env.custom_observation(goal=avoid_zone) for avoid_zone in avoid_zones]
            avoid_zones_vs = []
            for idx in range(len(avoid_zones)):
                x_actor, x_critic = model.policy.mlp_extractor(torch.as_tensor(avoid_zones_ob[idx]).float().to(device))
                avoid_zones_vs.append(model.policy.value_net(x_critic).detach().cpu())
            avoid_zones_vs = np.array(avoid_zones_vs)

            dangerous_zone_indices = np.argwhere(avoid_zones_vs > value_threshold)
            if dangerous_zone_indices.size > 0:

                # NOTE: v1 stragety, comobine v0 actions for every dangerous zones
                x_actor, x_critic = model.policy.mlp_extractor(torch.as_tensor(ob).float().to(device))
                goal_v = model.policy.value_net(x_critic)

                safe_actions = []
                goal_reaching_actions = []
                for index in dangerous_zone_indices:
                    index = int(index)
                    v = avoid_zones_vs[index]
                    if float(v) > goal_v[0].item():
                        # safe_action
                        action_distribution = model.policy.get_distribution(torch.from_numpy(avoid_zones_ob[index]).unsqueeze(dim=0).to(device))
                        action_probs = action_distribution.distribution.probs
                        safe_action = torch.argmin(action_probs, dim=1)
                        safe_actions.append(safe_action)

                        # (blocked) goal_reaching_aciton
                        avoid_zone_action, _states = model.predict(avoid_zones_ob[index], deterministic=True)
                        action_distribution = model.policy.get_distribution(torch.from_numpy(ob).unsqueeze(dim=0).to(device))
                        action_probs = action_distribution.distribution.probs
                        dangerous_mask = torch.ones(4).to(device)
                        dangerous_mask[avoid_zone_action] = 0
                        goal_reaching_action = torch.argmax(action_probs * dangerous_mask, dim=1)
                        goal_reaching_actions.append(goal_reaching_action)

                if len(safe_actions) > 0:
                    ob, reward, eval_done, info = env.step({
                        'action': safe_actions + goal_reaching_actions,
                        'distribution': [1/len(safe_actions) for _ in range(len(safe_actions))] * 2,
                    })
                elif len(safe_actions) == 0:
                    action, _states = model.predict(ob, deterministic=True)
                    ob, reward, eval_done, info = env.step(action)

            else:
                action, _states = model.predict(ob, deterministic=True)
                ob, reward, eval_done, info = env.step(action)

            if info['zone'] in avoid_zones:
                print('[Dangerous !][reach {} avoid {}][overlap with {}][seed][{}]'.format(goal_zone, avoid_zones, info['zone'], seed))
                return (ob, reward, eval_done, info), {'complete': False, 'dangerous': True, 'reach_zone': None}
            
            if info['zone'] == goal_zone:
                return (ob, reward, eval_done, info), {'complete': True, 'dangerous': False, 'reach_zone': goal_zone}

    return (ob, 0, True, {}), {'complete': False, 'dangerous': False, 'reach_zone': None}


def main(args):

    seed = args.seed
    temp = args.temp
    rl_model_path = args.rl_model_path
    eval_repeats = args.eval_repeats
    value_threshold = args.value_threshold
    device = torch.device(args.device)
    formula = args.formula
    subtask_function_version = args.subtask_function_version
    max_timesteps = args.max_timesteps

    model = PPO.load(rl_model_path, device=device)
    num_success = 0
    num_J_success = 0
    num_W_success = 0
    num_R_success = 0
    num_Y_success = 0
    num_dangerous = 0
    
    sampler = getLTLSampler('Until_1_2_1_1', ['J', 'W', 'R', 'Y'])
                                                                   
    if subtask_function_version == 'v0':
        subtask_function = ltl_subtask_v0
    elif subtask_function_version == 'v1':
        subtask_function = ltl_subtask_v1
    total_rewards = 0
    goals_representation = get_named_goal_vector()

    with torch.no_grad():

        for i in range(eval_repeats):

            random.seed(seed + i)

            raw_formula = sampler.sample()
            formula = reformat_ltl(raw_formula)
            ltl_args = get_ltl_args(formula=formula)
            graph = gltl2ba(ltl_args)
            path = SCC_Algorithm(graph=graph).search()
            GOALS, AVOID_ZONES = parse_ltl_path(path['ltl'])
            
            print('+'*80)
            print('[ITERATION][{}]'.format(i))
            print('[FORMULA]', formula, '[GOALS]', GOALS, '[AVOID]', AVOID_ZONES)

            stage_index = 0
            env = RandomGoalLTLNormalEnv(
                env=gym.make('Zones-8-v1', map_seed=seed+i, timeout=10000000),  # NOTE: dummy timeout
                primitives_path='./models/primitives',
                goals_representation=goals_representation, 
                use_primitves=True,
                temperature=temp,
                device=device,
                max_timesteps=max_timesteps,
                debug=True,
            )
            env.reset()

            for stage_index in range(len(GOALS)):
                _, subtask_info = subtask_function(env, model, GOALS[stage_index], AVOID_ZONES[stage_index], value_threshold=value_threshold, seed=seed+i, device=device)
                if subtask_info['complete']:
                    if subtask_info['reach_zone'] == 'J':
                        num_J_success += 1
                    elif subtask_info['reach_zone'] == 'W':
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
            sys.stdout.flush()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temp', type=float, default=1.5)
    parser.add_argument('--max_timesteps', type=int, default=1000),
    parser.add_argument('--subtask_function_version', type=str, default='v0', choices=('v0', 'v1'))
    parser.add_argument('--rl_model_path', type=str, default='models/goal-conditioned/best_model_ppo_8')
    parser.add_argument('--eval_repeats', type=int, default=10)
    parser.add_argument('--value_threshold', type=float, default=0.85)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--formula', type=str, default='')
    
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(args)
