from spectrl.hierarchy.construction import automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.spec_compiler import ev, seq, choose, alw
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ars import HyperParams

from spectrl.examples.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM
from spectrl.envs.rooms import RoomsEnv

import os

num_iters = [50, 100, 200, 300, 400, 500]

# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    render = flags['render']
    env_num = flags['env_num']
    folder = flags['folder']
    itno = flags['itno']
    spec_num = flags['spec_num']

    log_info = []

    for i in num_iters:

        grid_params = GRID_PARAMS_LIST[env_num]

        hyperparams = HyperParams(30, i, 30, 15, 0.05, 0.3, 0.15)

        print('\n**** Learning Policy for Spec #{} in Env #{} ****'.format(spec_num, env_num))

        # Step 1: initialize system environment
        system = RoomsEnv(grid_params, START_ROOM[env_num], FINAL_ROOM[env_num])

        # Step 4: List of specs.
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

        # Step 3: construct abstract reachability graph
        _, abstract_reach = automaton_graph_from_spec(specs[spec_num])
        print('\n**** Abstract Graph ****')
        abstract_reach.pretty_print()

    #     # Step 5: Learn policy
    #     abstract_policy, nn_policies, stats = abstract_reach.learn_dijkstra_policy(
    #         system, hyperparams, res_model=None, max_steps=20, render=render,
    #         neg_inf=-100, safety_penalty=-1, num_samples=500)

    #     # Test policy
    #     hierarchical_policy = HierarchicalPolicy(
    #         abstract_policy, nn_policies, abstract_reach.abstract_graph, 2)
    #     final_env = ConstrainedEnv(system, abstract_reach, abstract_policy,
    #                                res_model=None, max_steps=MAX_TIMESTEPS[env_num])

    #     # Print statements
    #     _, prob = print_performance(final_env, hierarchical_policy, stateful_policy=True)
    #     print('\nTotal Sample Steps: {}'.format(stats[0]))
    #     print('Total Time Taken: {} mins'.format(stats[1]))
    #     print('Total Edges Learned: {}'.format(stats[2]))

    #     # Render
    #     if render:
    #         print('\nSimulation with learned policy...')
    #         get_rollout(final_env, hierarchical_policy, True, stateful_policy=True)

    #     logdir = os.path.join(folder, 'spec{}'.format(spec_num), 'hierarchy')
    #     if not os.path.exists(logdir):
    #         os.makedirs(logdir)
    #     log_info.append([stats[0], stats[1], prob])

    # save_log_info(log_info, itno, logdir)
