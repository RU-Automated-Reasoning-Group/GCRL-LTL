import doodad as dd
import gcsl.doodad_utils as dd_utils
import sys
import matplotlib.pyplot as plt
import gym


def run(output_dir='/tmp', env_name='pointmass_empty', gpu=True, seed=0, **kwargs):

    import gym
    import numpy as np
    from rlutil.logging import log_utils, logger

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu

    # Envs

    from gcsl import envs

    # Algo
    from gcsl.algo import buffer, gcsl, variants, networks
    from RRT_star.algo import RRTStar_GCSL
    from RRT_star.Graph import Graph
    from RRT_star.v_function_core import value_policy
    import pickle
    from spectrl.examples.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM
    from spectrl.envs.rooms import RoomsEnv
    from spectrl.main.spec_compiler import ev, seq, choose, alw
    from spectrl.hierarchy.construction import automaton_graph_from_spec

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    np.random.seed(seed)

    # get ant env
    env = envs.create_env(env_name)
    env.print_maze_infos()
    env_for_checking = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, env_for_checking, policy, replay_buffer, gcsl_kwargs = variants.get_params(
        env, env_for_checking, env_params)
    RRT_star_tree = Graph((0, 0), (0, 27), algo='Dijkstra')
    file_path = '/root/code/gcsl_ant/data/example/' + env_name + '/rrt_star/' + env_name + \
        '_test5_afterfunetuning_SL[400and100]_correctrandomposition/'
    policy_filename = file_path + 'policy.pkl'

    # load main policy trained by SL
    policy.load_state_dict(torch.load(policy_filename))
    policy.eval()

    # load value policy
    valuepolicy = value_policy(env)
    valuepolicyfilepath = file_path + 'value_policy.pkl'
    valuepolicy.load_policy(valuepolicyfilepath)
    policy = policy.cuda()

    algo = RRTStar_GCSL(
        env_name,
        env,
        env_for_checking,
        RRT_star_tree,
        policy,
        valuepolicy,
        replay_buffer,
        **gcsl_kwargs
    )

    # get LTL specs
    env_num = 2
    # For 9rooms env: {spec_number} is either 3, 4, 5, 6 or 7 corresponding to specs 1 to 5 in the paper.
    spec_num = 3
    grid_params = GRID_PARAMS_LIST[env_num]

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

    spec1 = ev(grid_params.in_room(bottomright))
    # spec2 = ev(grid_params.in_room(bottomright))
    spec2 = seq(ev(grid_params.in_room(bottomright)),
                ev(grid_params.in_room(FINAL_ROOM[env_num])))

    # Goto destination, return to initial
    spec3 = seq(ev(grid_params.in_room(topleft)), ev(
        grid_params.in_room(START_ROOM[env_num])))
    # Choose between top-right and bottom-left blocks (Same difficulty - learns 3/4 edges)
    spec4 = choose(ev(grid_params.in_room(bottomright)),
                   ev(grid_params.in_room(topleft)))
    # Choose between top-right and bottom-left, then go to Final state (top-right).
    # Only one path is possible (learns 5/5 edges. Should have a bad edge)
    spec5 = seq(choose(ev(grid_params.in_room(bottomright)),
                       ev(grid_params.in_room(topleft))),
                ev(grid_params.in_room(FINAL_ROOM[env_num])))
    # Add obsacle towards topleft
    spec6 = alw(grid_params.avoid_center((1, 0)),
                ev(grid_params.in_room(topleft)))
    # Either go to top-left or bottom-right. obstacle on the way to top-left.
    # Then, go to Final state. Only one route is possible

    spec7 = seq(choose(alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room(topleft))),
                       ev(grid_params.in_room(bottomright))),
                ev(grid_params.in_room(FINAL_ROOM[env_num])))
    # spec7 = seq(choose(ev(grid_params.in_room(topleft)),
    #                    ev(grid_params.in_room(bottomright))),
    #             ev(grid_params.in_room(FINAL_ROOM[env_num])))

    specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6, spec7]

    # Step 3: construct abstract reachability graph
    _, abstract_reach = automaton_graph_from_spec(specs[spec_num])
    print('\n**** Abstract Graph ****')
    abstract_reach.pretty_print()

    # get LTL graph
    abstract_LTL_graph = abstract_reach.abstract_graph
    abstract_final_vertices = abstract_reach.final_vertices

    exp_prefix = 'example/%s/rrt_star/' % (env_name,)
    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.visualize_evaluate_policy_with_LTL(
            abstract_LTL_graph, abstract_final_vertices, eval_episodes=50, greedy=True, prefix='Eval', env_type=env_name)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    env_name = str(sys.argv[1])
    # assert env_name in ['antumaze', 'antfall', 'antpush', 'antfourrooms']
    params = {
        'seed': [0],
        'env_name': [env_name],
        'gpu': [True],
    }
    dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
