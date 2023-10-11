import argparse
import random

import torch
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList

from rl import GCPPO, PolicyCheckpointCallback, SimpleEvalCallback
from envs import ZonesEnv, ZoneRandomGoalContinualEnv
from envs.utils import get_zone_vector


def main(args):

    device = torch.device(args.device)
    timeout = args.timeout
    total_timesteps = args.total_timesteps
    num_cpus = args.num_cpus
    seed = args.seed
    exp_name = args.exp_name
    execution_mode = args.execution_mode

    env_fn = lambda: ZoneRandomGoalContinualEnv(
        env=gym.make('Zones-8-v0', timeout=timeout), 
        primitives_path='models/primitives', 
        zones_representation=get_zone_vector(),
        use_primitves=True if execution_mode == 'primitives' else False,
        rewards=[0, 1],
        device=device,
        reset_continual=True,
    )

    env = make_vec_env(env_fn, n_envs=num_cpus, seed=seed, vec_env_cls=SubprocVecEnv)
    model = GCPPO(
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

    log_path = 'logs/ppo/{}/'.format(exp_name)
    new_logger = configure(log_path, ['stdout', 'csv'])
    model.set_logger(new_logger)

    eval_log_path = 'logs/ppo/{}/'.format(exp_name)
    eval_env_fn = lambda: ZoneRandomGoalContinualEnv(
        env=gym.make('Zones-8-v0', timeout=1000),
        primitives_path='models/primitives',
        zones_representation=get_zone_vector(),
        use_primitves=True if execution_mode == 'primitives' else False,
        rewards=[-0.001, 1],
        device=device,
        reset_continual=False,
    )
    eval_env = make_vec_env(eval_env_fn)
    eval_callback = SimpleEvalCallback(
        eval_env=eval_env, 
        # NOTE: save GCPPO model directly is not available
        best_model_save_path=eval_log_path,
        log_path=eval_log_path,
        eval_freq=100000/num_cpus,
        n_eval_episodes=20,
        deterministic=True,
    )
    
    policy_checkpoint_callback = PolicyCheckpointCallback(
        save_freq=100000/num_cpus,
        save_path='logs/ppo_checkpoint/{}/'.format(exp_name),
        name_prefix='gc_ppo',
    )

    callback = CallbackList([eval_callback, policy_checkpoint_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--total_timesteps', type=int, default=1.5e7)
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--exp_name', type=str, default='continual_exp')
    parser.add_argument('--execution_mode', type=str, default='primitives', choices=('primitives'))
    
    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(args)
