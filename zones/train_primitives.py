import argparse
import random

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from envs import ZonesEnv
from ltl_wrappers import ZonePrimitiveEnv


def main(args):

    direction = args.direction
    timeout = args.timeout
    total_timesteps = args.total_timesteps
    num_cpu = args.num_cpu
    seed = args.seed

    env_fn = lambda : ZonePrimitiveEnv(env=ZonesEnv(zones=[], use_fixed_map=False, timeout=timeout, config={}, walled=False), direction=direction)
    env = make_vec_env(env_fn, n_envs=num_cpu, seed=seed, vec_env_cls=SubprocVecEnv)
    
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    model.save('/models/primitives/{}'.format(direction))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', type=str, default='pos_x', choices=('pos_x', 'neg_x', 'pos_y', 'neg_y'))
    parser.add_argument('--total_timesteps', type=int, default=1e6)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--num_cpu', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    
    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(args)
