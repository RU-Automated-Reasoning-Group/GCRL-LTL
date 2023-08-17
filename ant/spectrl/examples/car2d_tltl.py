from spectrl.main.learning import QSRewardEnv, Resource_Model
from spectrl.main.spec_compiler import ev, alw, seq, choose
from spectrl.util.rl import print_performance, get_rollout
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.envs.car2d import VC_Env
from spectrl.rl.ars import ARSParams, NNPolicy, NNParams, ars
from numpy import linalg as LA

import os
import numpy as np


# Define the resource model
# Fuel consumption proportional to distance from x-axis and the velocity
# sys_state: np.array(2)
# res_state: np.array(1)
# sys_action: np.array(2)
def resource_delta(sys_state, res_state, sys_action):
    return np.array([res_state[0] - 0.1 * abs(sys_state[0]) * LA.norm(sys_action)])


# Define the specification
# 1. Relevant atomic predicates:
# a. Reach predicate
#    goal: np.array(2), err: float
def reach(goal, err):
    def predicate(sys_state, res_state):
        return (min([sys_state[0] - goal[0],
                     goal[0] - sys_state[0],
                     sys_state[1] - goal[1],
                     goal[1] - sys_state[1]]) + err)
    return predicate


# b. Avoid predicate
#    obstacle: np.array(4): [x_min, y_min, x_max, y_max]
def avoid(obstacle):
    def predicate(sys_state, res_state):
        return max([obstacle[0] - sys_state[0],
                    obstacle[1] - sys_state[1],
                    sys_state[0] - obstacle[2],
                    sys_state[1] - obstacle[3]])
    return predicate


def have_fuel(sys_state, res_state):
    return res_state[0]


# Goals and obstacles
gtop = np.array([5.0, 10.0])
gbot = np.array([5.0, 0.0])
gright = np.array([10.0, 0.0])
gcorner = np.array([10.0, 10.0])
gcorner2 = np.array([0.0, 10.0])
origin = np.array([0.0, 0.0])
obs = np.array([4.0, 4.0, 6.0, 6.0])

# Specifications
spec1 = alw(avoid(obs), ev(reach(gtop, 1.0)))
spec2 = alw(avoid(obs), alw(have_fuel, ev(reach(gtop, 1.0))))
spec3 = seq(alw(avoid(obs), ev(reach(gtop, 1.0))),
            alw(avoid(obs), ev(reach(gbot, 1.0))))
spec4 = seq(choose(alw(avoid(obs), ev(reach(gtop, 1.0))), alw(avoid(obs), ev(reach(gright, 1.0)))),
            alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec5 = seq(spec3, alw(avoid(obs), ev(reach(gright, 1.0))))
spec6 = seq(spec5, alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec7 = seq(spec6, alw(avoid(obs), ev(reach(origin, 1.0))))

gt1 = np.array([3.0, 4.0])
gt2 = np.array([6.0, 0.0])
gfinal = np.array([3.0, 7.0])
gfinal2 = np.array([7.0, 4.0])

spec8 = seq(choose(alw(avoid(obs), ev(reach(gt2, 1.0))), alw(avoid(obs), ev(reach(gt1, 1.0)))),
            alw(avoid(obs), ev(reach(gfinal, 1.0))))

spec9 = seq(choose(alw(avoid(obs), ev(reach(gt2, 1.0))), alw(avoid(obs), ev(reach(gt1, 1.0)))),
            alw(avoid(obs), ev(reach(gfinal2, 1.0))))


spec10 = choose(alw(avoid(obs), ev(reach(gt1, 1.0))),
                alw(avoid(obs), ev(reach(gt2, 1.0))))


specs = [spec1, spec2, spec3, spec4, spec5, spec6, spec7, spec8, spec9, spec10]

params_list = [ARSParams(1500, 20, 8, 0.04, 0.4, 0.4, 10),
               ARSParams(1500, 20, 8, 0.04, 0.4, 0.2, 10),
               ARSParams(1500, 20, 8, 0.04, 0.4, 0.2, 10),
               ARSParams(1500, 20, 8, 0.04, 0.5, 0.2, 10),
               ARSParams(2000, 30, 10, 0.04, 0.6, 0.2, 10),
               ARSParams(1500, 30, 10, 0.04, 0.8, 0.2, 10),
               ARSParams(3000, 30, 10, 0.04, 0.3, 0.2, 10),
               ARSParams(3000, 30, 10, 0.04, 0.3, 0.2, 10),
               ARSParams(3000, 30, 10, 0.04, 0.3, 0.2, 10),
               ARSParams(2000, 30, 10, 0.04, 0.3, 0.2, 10),
               ARSParams(2000, 30, 10, 0.04, 0.3, 0.2, 10),
               ARSParams(2000, 30, 10, 0.04, 0.3, 0.2, 10)]


fuel = [False, True, False, False, False, False, False, False, False, False]


# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    itno = flags['itno']
    folder = flags['folder']
    spec_num = flags['spec_num']
    render = flags['render']

    print('**** Learning Policy for Spec {} ****'.format(spec_num))

    # Step 1: initialize system environment
    time_limit = 40
    if spec_num >= 5:
        time_limit = 80
    system = VC_Env(time_limit, std=0.05)

    # Step 2 (optional): construct resource model
    resource = Resource_Model(2, 2, 1, np.array([7.0]), resource_delta)
    if not fuel[spec_num]:
        resource = None

    # Step 3: construct product MDP
    env = QSRewardEnv(system, specs[spec_num], resource)

    # Step 4: Set hyper parameters
    params = params_list[spec_num]

    # Step 5: Learn policy
    policy = NNPolicy(NNParams(env, 50))
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
