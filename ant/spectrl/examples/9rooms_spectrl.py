from spectrl.main.learning import ProductMDP, Resource_Model, learn_policy, HyperParams
# from spectrl.hierarchy.construction import automaton_graph_from_spec
# from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.spec_compiler import ev, seq, choose, alw
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout
# from spectrl.rl.ars import HyperParams
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


lb = [10., 20., 10., 10., 10., 9., 9., 9., 9., 9.]

fuel = [False, False, False, False, False, False, False, False, False]


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

    # No. of steps for 3*3 per spec. 20 steps for each edge on a path.

    stepsrequired = [50, 50, 50, 50, 50, 70, 50, 70]
    # Step 1: initialize system environment
    # system = RoomsEnv(grid_params, START_ROOM[env_num],
    #                  FINAL_ROOM[env_num], max_timesteps=MAX_TIMESTEPS[env_num])

    system = RoomsEnv(grid_params, START_ROOM[env_num],
                      FINAL_ROOM[env_num], max_timesteps=stepsrequired[spec_num])

    # Step 2 (optional): construct resource model
    resource = Resource_Model(2, 2, 1, np.array([7.0]), resource_delta)

    # Step 3: List of specs.
    if env_num == 2:
        bottomright = (0, 2)
        topleft = (2, 0)
    if env_num == 3 or env_num == 4:
        bottomright = (0, 3)
        topleft = (3, 0)

    # test specs
    spec0 = ev(grid_params.in_room(FINAL_ROOM[env_num]))
    spec1 = seq(ev(grid_params.in_room(FINAL_ROOM[env_num])), ev(
        grid_params.in_room(START_ROOM[env_num])))
    spec2 = ev(grid_params.in_room(topleft))

    # Goto destination, return to initial
    spec3 = seq(ev(grid_params.in_room(topleft)), ev(grid_params.in_room(START_ROOM[env_num])))
    # Choose between top-right and bottom-left blocks (Same difficulty - learns 3/4 edges)
    spec4 = choose(ev(grid_params.in_room(bottomright)),
                   ev(grid_params.in_room(topleft)))
    # Choose between top-right and bottom-left, then go to Final state (top-right).
    # Only one path is possible (learns 5/5 edges. Should have a bad edge)
    spec5 = seq(choose(ev(grid_params.in_room(bottomright)),
                       ev(grid_params.in_room(topleft))),
                ev(grid_params.in_room(FINAL_ROOM[env_num])))
    # Add obsacle towards topleft
    spec6 = alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room(topleft)))
    # Either go to top-left or bottom-right. obstacle on the way to top-left.
    # Then, go to Final state. Only one route is possible
    spec7 = seq(choose(alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room(topleft))),
                       ev(grid_params.in_room(bottomright))),
                ev(grid_params.in_room(FINAL_ROOM[env_num])))

    specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6, spec7]

    # Step 4: construct product MDP
    env = ProductMDP(system, 2, specs[spec_num], 0.0,
                     lb[spec_num], res_model=resource) if fuel[spec_num] else ProductMDP(
                         system, 2, specs[spec_num], 0.0, lb[spec_num])

    # Step 5: Hyperparameters
    hparameterlist = [1000, 1000, 1000, 1000, 1000, 1500, 1000, 1000]
    hyperparams = HyperParams(30, 1, hparameterlist[spec_num], 30, 8, 0.05, 0.3, 0.1)

    # Step 6: Learn policy
    policy, log_info = learn_policy(env, hyperparams)

    # Save policy and log information
    logdir = os.path.join(folder, 'spec{}'.format(spec_num), 'spectrl')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    save_log_info(log_info, itno, logdir)
    # save_object('policy', policy, itno, logdir)

    # Print rollout and performance
    print_performance(env, policy)
    if render:
        get_rollout(env, policy, True)
