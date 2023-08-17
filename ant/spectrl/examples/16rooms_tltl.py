from spectrl.main.learning import Resource_Model, QSRewardEnv
from spectrl.main.spec_compiler import ev, seq, choose, alw
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ars import ars, NNParams, NNPolicy, ARSParams
from numpy import linalg as LA

import os
import numpy as np

from spectrl.examples.rooms_envs import GRID_PARAMS_LIST, START_ROOM, FINAL_ROOM
from spectrl.envs.rooms import RoomsEnv


# Define the resource model
# Fuel consumption proportional to distance from x-axis and the velocity
# sys_state: np.array(2)
# res_state: np.array(1)
# sys_action: np.array(2)
def resource_delta(sys_state, res_state, sys_action):
    return np.array([res_state[0] - 0.1 * abs(sys_state[0]) * LA.norm(sys_action)])


lb = [10., 20., 10., 10., 10., 9., 9., 9., 9., 9., 9., 9., 9., 9.]

fuel = [False, False, False, False, False, False, False,
        False, False, False, False, False, False, False]


# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    render = flags['render']
    env_num = flags['env_num']
    folder = flags['folder']
    itno = flags['itno']
    spec_num = flags['spec_num']

    grid_params = GRID_PARAMS_LIST[env_num]

    print('\n**** Learning Policy for Spec #{} in Env #{} ****'.format(spec_num, env_num))

    # Step 1: initialize system environment

    stepsrequired = [50, 50, 50, 50, 70, 70, 50,
                     50, 50, 50, 70, 110, 150, 190]
    system = RoomsEnv(grid_params, START_ROOM[env_num],
                      FINAL_ROOM[env_num], max_timesteps=stepsrequired[spec_num])

    # Step 2 (optional): construct resource model
    resource = Resource_Model(2, 2, 1, np.array([7.0]), resource_delta)
    if not fuel[spec_num]:
        resource = None

    bottomright = (0, 3)
    topleft = (3, 0)

    # test specs
    spec0 = ev(grid_params.in_room((1, 0)))  # 100 iterations works
    spec1 = ev(grid_params.in_room((2, 0)))  # 200 iterations works
    spec2 = ev(grid_params.in_room(topleft))  # 1000 iterations not enough
    spec3 = alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room((2, 0))))  # 100 works
    spec4 = seq(ev(grid_params.in_room((1, 0))), ev(grid_params.in_room(topleft)))  # 200 works
    spec5 = seq(ev(grid_params.in_room((2, 0))), ev(grid_params.in_room((2, 2))))  # 200 works
    spec6 = alw(grid_params.avoid_center((1, 0)), ev(
        grid_params.in_room(topleft)))  # 400 doesn't work
    spec7 = alw(grid_params.avoid_center((2, 0)), ev(grid_params.in_room(topleft)))
    spec8 = choose(ev(grid_params.in_room(topleft)),
                   ev(grid_params.in_room(bottomright)))

    spec9 = choose(alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room((2, 0)))),
                   ev(grid_params.in_room((0, 2))))
    spec10 = seq(spec9, ev(grid_params.in_room((2, 2))))

    spec10part1 = choose(ev(grid_params.in_room((2, 1))),
                         ev(grid_params.in_room((3, 2))))
    spec10part2 = seq(spec10part1,
                      ev(grid_params.in_room((3, 1))))
    spec11 = seq(spec10, spec10part2)

    spec11part1 = choose(ev(grid_params.in_room((1, 1))),
                         ev(grid_params.in_room((3, 3))))
    spec11part2 = seq(spec11part1,
                      alw(grid_params.avoid_center((2, 3)), ev(grid_params.in_room((1, 3)))))
    spec12 = seq(spec11, spec11part2)

    spec12part1 = choose(ev(grid_params.in_room((1, 1))),
                         ev(grid_params.in_room((0, 3))))
    spec12part2 = seq(spec12part1, ev(grid_params.in_room((0, 1))))
    spec13 = seq(spec12, spec12part2)

    specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6,
             spec7, spec8, spec9, spec10, spec11, spec12, spec13]

    # Step 4: construct product MDP
    env = QSRewardEnv(system, specs[spec_num], res_model=resource)

    # Step 5: Hyperparameters
    hparameterlist = [1000, 1000, 1000, 1000, 1000, 1000, 1000,
                      1000, 1000, 1500, 1000, 1500, 1500, 1000]

    hyperparams = ARSParams(hparameterlist[spec_num], 30, 10, 0.03, 0.4, 0.2, 10)

    # Step 6: Learn policy
    policy = NNPolicy(NNParams(env, 50))
    log_info = ars(env, policy, hyperparams, cum_reward=True)

    # Save policy and log information
    logdir = os.path.join(folder, 'spec{}'.format(spec_num), 'tltl')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    save_log_info(log_info, itno, logdir)
    # save_object('policy', policy, itno, logdir)

    # Print rollout and performance
    print_performance(env, policy)
    if render:
        get_rollout(env, policy, True)
