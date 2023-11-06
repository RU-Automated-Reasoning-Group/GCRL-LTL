import argparse
import random

import torch
import numpy as np
import gym
from stable_baselines3 import PPO

from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset

ZONE_OBS_DIM = 24


def get_goal_value(state, policy, zone_vector, device):
    goal_value = {'J': None, 'W': None, 'R': None, 'Y': None}
    for zone in zone_vector:
        if not np.allclose(state[-ZONE_OBS_DIM:], zone_vector[zone]):
            with torch.no_grad():
                obs = torch.from_numpy(np.concatenate((state[:-ZONE_OBS_DIM], zone_vector[zone]))).unsqueeze(dim=0).to(device)
                goal_value[zone] = policy.predict_values(obs)[0].item()
    
    return goal_value


def main(args):

    device = torch.device(args.device)
    timeout = args.timeout
    buffer_size = args.buffer_size
    model_path = args.model_path
    exp_name = args.exp_name
    
    model = PPO.load(model_path, device=device)
    env = ZoneRandomGoalEnv(
        env=gym.make('Zones-8-v0', timeout=timeout), 
        primitives_path='models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )
    
    # build dataset
    states, goal_values = [], []
    total_steps = 0
    with torch.no_grad():
        while total_steps < buffer_size:
            
            local_states, local_goal_values = [], []
            local_steps = 0
            eval_done = False
            
            obs = env.reset()
            while not eval_done and local_steps < timeout:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, eval_done, info = env.step(action)
                local_states.append(obs)
            
            if reward == 1:
                _local_states = []
                for state in local_states:
                    values = get_goal_value(state, model.policy, get_zone_vector(), device)
                    for g in values:
                        if values[g]:
                            _local_states.append(np.concatenate((state, get_zone_vector()[g]), axis=0))
                            local_goal_values.append(values[g])
                states += _local_states
                goal_values += local_goal_values
                assert len(_local_states) == len(local_goal_values)
            
                total_steps += len(local_goal_values)

    # save dataset
    dataset = TrajectoryBufferDataset(states=states, goal_values=goal_values)
    torch.save(dataset, 'datasets/{}.pt'.format(exp_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--buffer_size', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model_path', type=str, default='models/goal-conditioned/best_model_ppo_8')
    #parser.add_argument('--model_path', type=str, default='models/gc_ppo_policy')
    parser.add_argument('--exp_name', type=str, default='my_traj_dataset')
    parser.add_argument('--execution_mode', type=str, default='primitives', choices=('primitives'))
    
    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(args)