from spectrl.main.learning import QSRewardEnv
from spectrl.main.spec_compiler import ev, seq, choose
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout, ObservationWrapper
from spectrl.envs.fetch import FetchPickAndPlaceEnv
from spectrl.rl.ars import ars, NNParams, ARSParams, NNPolicy
from numpy import linalg as LA

import numpy as np
import os


def grip_near_object(err):
    def predicate(sys_state, res_state):
        dist = sys_state[:3] - (sys_state[3:6] + np.array([0., 0., 0.065]))
        dist = np.concatenate([dist, [sys_state[9] + sys_state[10] - 0.1]])
        return -LA.norm(dist) + err
    return predicate


def hold_object(err):
    def predicate(sys_state, res_state):
        dist = sys_state[:3] - sys_state[3:6]
        dist2 = np.concatenate([dist, [sys_state[9] + sys_state[10] - 0.045]])
        return -LA.norm(dist2) + err
    return predicate


def object_in_air(sys_state, res_state):
    return sys_state[5] - 0.45


def object_at_goal(err):
    def predicate(sys_state, res_state):
        dist = np.concatenate([sys_state[-3:], [sys_state[9] + sys_state[10] - 0.045]])
        return -LA.norm(dist) + err
    return predicate


def gripper_reach(goal, err):
    '''
    goal: numpy array of dim (3,)
    '''
    def predicate(sys_state, res_state):
        return -LA.norm(sys_state[:3] - goal) + err
    return predicate


def object_reach(goal, err):
    '''
    goal: numpy array of dim (3,)
    '''
    def predicate(sys_state, res_state):
        return -LA.norm(sys_state[3:6] - goal) + err
    return predicate


above_corner1 = np.array([1.15, 1.0, 0.465])
above_corner2 = np.array([1.45, 1.0, 0.465])
corner1 = np.array([1.15, 1.0, 0.425])
corner2 = np.array([1.50, 1.05, 0.425])

# Specifications
spec1 = ev(grip_near_object(0.03))
spec2 = seq(spec1, ev(hold_object(0.03)))
spec3 = seq(spec2, ev(object_at_goal(0.05)))
spec4 = seq(seq(spec2, ev(object_in_air)), ev(object_at_goal(0.05)))
spec5 = seq(seq(spec2, ev(object_in_air)), ev(object_reach(above_corner1, 0.05)))
spec6 = seq(seq(spec2, ev(object_in_air)),
            choose(seq(ev(object_reach(above_corner1, 0.05)), ev(object_reach(corner1, 0.05))),
                   seq(ev(object_reach(above_corner2, 0.05)), ev(object_reach(corner2, 0.01)))))

specs = [spec1, spec2, spec3, spec4, spec5, spec6]

# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    render = flags['render']
    folder = flags['folder']
    itno = flags['itno']
    spec_num = flags['spec_num']

    env = ObservationWrapper(FetchPickAndPlaceEnv(), ['observation', 'desired_goal'],
                             relative=(('desired_goal', 0, 3), ('observation', 3, 6)),
                             max_timesteps=150)
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    params_list = [ARSParams(500, 50, 20, 0.01, 0.5, 0.3, 10),
                   ARSParams(500, 50, 20, 0.01, 0.5, 0.3, 10),
                   ARSParams(500, 50, 20, 0.01, 0.5, 0.3, 10),
                   ARSParams(500, 50, 20, 0.01, 0.5, 0.3, 10),
                   ARSParams(500, 50, 20, 0.01, 0.5, 0.3, 10),
                   ARSParams(800, 50, 20, 0.01, 0.5, 0.3, 10)]

    print('\n**** Learning Policy for Spec {} ****'.format(spec_num))

    # Step 3: construct product MDP
    env = QSRewardEnv(env, specs[spec_num])

    # Step 4: Set hyper parameters
    params = params_list[spec_num]

    # Step 5: Learn policy
    policy = NNPolicy(NNParams(env, 300))
    log_info = ars(env, policy, params, cum_reward=True)

    # Save policy and log information
    logdir = os.path.join(folder, 'spec{}'.format(spec_num), 'tltl')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    save_log_info(log_info, itno, logdir)
    # save_object('policy', policy, itno, logdir)

    # Print rollout and performance
    print_performance(env, policy)
    if render:
        rollout = get_rollout(env, policy, True)
