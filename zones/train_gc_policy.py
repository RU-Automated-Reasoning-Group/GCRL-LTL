import argparse
import random

import torch
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from envs import ZonesEnv
from ltl_wrappers import RandomGoalLTLNormalEnv
from utils import get_named_goal_vector


def main(args):

    device = torch.device(args.device)
    timeout = args.timeout
    total_timesteps = args.total_timesteps
    num_cpus = args.num_cpus
    seed = args.seed
    algo = args.algo
    exp_name = args.exp_name
    execution_mode = args.execution_mode

    env_fn = lambda: RandomGoalLTLNormalEnv(
        env=gym.make('Zones-5-v0', timeout=timeout), 
        primitives_path='./models/primitives', 
        goals_representation=get_named_goal_vector(),
        use_primitves=True if execution_mode == 'primitives' else False,
        rewards=[0, 1],
        device=device,
    )

    if algo == 'ppo':
        env = make_vec_env(env_fn, n_envs=num_cpus, seed=seed, vec_env_cls=SubprocVecEnv)
        model = PPO(
            policy='MlpPolicy',
            policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[512, 1024, 256]),
            env=env,
            verbose=1,
            learning_rate=0.0003,
            gamma=0.998,
            n_epochs=10,
            n_steps=int(50000/num_cpus),
            batch_size=1000,
            ent_coef=0.003,
            device=device,
        )

        log_path = "./logs/ppo/{}/".format(exp_name)
        new_logger = configure(log_path, ["stdout", "csv"])
        model.set_logger(new_logger)

        eval_log_path = "./logs/ppo_eval/{}/".format(exp_name)
        eval_env_fn = lambda: RandomGoalLTLNormalEnv(
            env=gym.make('Zones-5-v0', timeout=1000), 
            primitives_path='./models/primitives', 
            goals_representation=get_named_goal_vector(),
            use_primitves=True if execution_mode == 'primitives' else False,
            rewards=[-0.001, 1],
        )
        eval_env = make_vec_env(eval_env_fn)
        eval_callback = EvalCallback(
            eval_env=eval_env, 
            best_model_save_path=eval_log_path,
            log_path=eval_log_path, 
            eval_freq=100000/num_cpus,
            n_eval_episodes=40,
            deterministic=True,
        )
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save('./{}_{}_model'.format(algo, seed))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--total_timesteps', type=int, default=1.5e7)
    parser.add_argument('--algo', type=str, default='ppo', choices=('ppo'))
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--execution_mode', type=str, default='primitives', choices=('primitives'))
    
    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(args)
