import doodad as dd
import algo.doodad_utils as dd_utils
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
    from algo import envs
    from algo.envs.env_utils import DiscretizedActionEnv

    # Algo
    from algo.gcsl_utils import buffer, variants, networks
    from algo.GCSL_Graph import GCSL_Graph
    from algo.graph_utils import Graph
    from algo.graph_utils.v_function_core import value_policy
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
    torch.set_num_threads(8)

    # get ant env
    env = envs.create_env(env_name)
    env.print_maze_infos()
    env_for_checking = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, env_for_checking, policy, replay_buffer, gcsl_kwargs = variants.get_params(
        env, env_for_checking, env_params)
    file_path = '/root/code/gcsl_ant/data/example/' + env_name + '/rrt_star/' + env_name + '_test10_finetune1e6(center)_redo2_SL/'
        # '_test10_finetune1e6(center)_SLcenterSG_3x256_batch512[1000and200]/'
    policy_filename = file_path + 'policy.pkl'

    # load main policy trained by SL
    policy.load_state_dict(torch.load(policy_filename))
    policy.eval()

    # load value policy and graph
    valuepolicy = value_policy(env)
    valuepolicy_filename = file_path + 'value_policy.pkl'
    valuepolicy.load_policy(valuepolicy_filename)
    graph_filename = file_path + 'graph.pkl'
    with open(graph_filename, 'rb') as f:
        graph = pickle.load(f)

    graph.set_algo('Dijkstra')
    policy = policy.cuda()

    algo = GCSL_Graph(
        env_name,
        env,
        env_for_checking,
        graph,
        policy,
        valuepolicy,
        replay_buffer,
        **gcsl_kwargs
    )

    # get LTL specs
    env_num = 4
    # For 16rooms env: {spec_number} is either 9, 10, 11, 12 or 13 corresponding to specs 1 to 5 in the paper.
    spec_num = 9
    
    # spec_nums_set = [13]
    spec_nums_set = [kwargs['spec_num']]
    spec_nums_set = [9, 10, 11, 12, 13]
    
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
    spec0 = ev(grid_params.in_room((1, 0)))  #
    spec1 = ev(grid_params.in_room((2, 0)))  #
    spec2 = ev(grid_params.in_room(topleft))  #
    spec3 = alw(grid_params.avoid_center((1, 0)),
                ev(grid_params.in_room((2, 0))))  #
    spec4 = seq(ev(grid_params.in_room((1, 0))),
                ev(grid_params.in_room(topleft)))  #
    spec5 = seq(ev(grid_params.in_room((2, 0))),
                ev(grid_params.in_room((2, 2))))  #
    spec6 = alw(grid_params.avoid_center((1, 0)),
                ev(grid_params.in_room(topleft)))  #
    spec7 = alw(grid_params.avoid_center((2, 0)),
                ev(grid_params.in_room(topleft)))

    spec8 = seq(ev(grid_params.in_room((3, 2))),
                ev(grid_params.in_room(topleft)))

    # \phi1
    # spec9 = choose(alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room((2, 0)))),
    #                ev(grid_params.in_room((0, 2))))
    spec9 = choose(ev(grid_params.in_room((2, 0))),
                   ev(grid_params.in_room((0, 2))))
    # spec9 = ev(grid_params.in_room((0, 2)))

    # \phi2
    spec10 = seq(spec9, ev(grid_params.in_room((2, 2))))

    spec10part1 = choose(ev(grid_params.in_room((2, 1))),
                         ev(grid_params.in_room((3, 2))))
    spec10part2 = seq(spec10part1,
                      ev(grid_params.in_room((3, 1))))
    # \phi3
    spec11 = seq(spec10, spec10part2)

    spec11part1 = choose(ev(grid_params.in_room((1, 1))),
                         ev(grid_params.in_room((3, 3))))
    # spec11part2 = seq(spec11part1,
    #                   alw(grid_params.avoid_center((2, 3)), ev(grid_params.in_room((1, 3)))))
    spec11part2 = seq(spec11part1, ev(grid_params.in_room((1, 3))))
    # spec11part2easy = ev(grid_params.in_room((1,3)))
    # \phi4
    spec12 = seq(spec11, spec11part2)

    spec12part1 = choose(ev(grid_params.in_room((1, 1))),
                         ev(grid_params.in_room((0, 3))))
    spec12part2 = seq(spec12part1, ev(grid_params.in_room((0, 1))))
    # \phi5
    spec13 = seq(spec12, spec12part2)

    spec14part1 = seq(choose(ev(grid_params.in_room((2, 0))), ev(
        grid_params.in_room((0, 1)))), ev(grid_params.in_room((2, 1))))
    spec14part2 = seq(choose(ev(grid_params.in_room((2, 2))), ev(
        grid_params.in_room((1, 1)))), ev(grid_params.in_room((1, 2))))
    spec14part3 = seq(choose(ev(grid_params.in_room((3, 2))), ev(
        grid_params.in_room((1, 3)))), ev(grid_params.in_room((3, 3))))
    # \phi6
    spec14 = seq(seq(spec14part1, spec14part2), spec14part3)

    spec15part1 = seq(choose(ev(grid_params.in_room((2, 0))), ev(
        grid_params.in_room((0, 2)))), ev(grid_params.in_room((2, 2))))
    spec15part2 = seq(choose(ev(grid_params.in_room((2, 1))), ev(
        grid_params.in_room((1, 2)))), ev(grid_params.in_room((1, 1))))
    spec15part3 = seq(choose(ev(grid_params.in_room((0, 1))), ev(
        grid_params.in_room((1, 3)))), ev(grid_params.in_room((0, 3))))
    # \phi7
    spec15 = seq(seq(spec15part1, spec15part2), spec15part3)

    spec16part1 = seq(choose(ev(grid_params.in_room((2, 0))), ev(
        grid_params.in_room((0, 2)))), ev(grid_params.in_room((2, 2))))
    spec16part2 = seq(seq(seq(ev(grid_params.in_room((2, 1))), ev(grid_params.in_room(
        (1, 1)))), ev(grid_params.in_room((1, 2)))), ev(grid_params.in_room((2, 2))))
    spec16part3 = seq(seq(seq(ev(grid_params.in_room((2, 1))), ev(grid_params.in_room(
        (1, 1)))), ev(grid_params.in_room((1, 2)))), ev(grid_params.in_room((2, 2))))
    # \phi8
    spec16 = seq(seq(spec16part1, spec16part2), spec16part3)

    # Additional specifications after NeurIPS2023 review
    # spec17
    spec17part1 = seq(ev(grid_params.in_room((0, 2))),
                      ev(grid_params.in_room((1, 2))))
    # big loop
    spec17part2 = ev(grid_params.in_room((2, 2)))
    spec17part3 = seq(seq(seq(ev(grid_params.in_room((3, 2))), ev(grid_params.in_room(
        (3, 0)))), ev(grid_params.in_room((2, 0)))), ev(grid_params.in_room((2, 2))))
    for i in range(5):
        spec17part2 = seq(spec17part2, spec17part3)

    # small loop
    spec17part4 = seq(seq(seq(ev(grid_params.in_room((1, 3))), ev(grid_params.in_room(
        (0, 3)))), ev(grid_params.in_room((0, 2)))), ev(grid_params.in_room((1, 2))))
    spec17part5 = seq(spec17part4, spec17part4)
    for i in range(3):
        spec17part5 = seq(spec17part5, spec17part4)

    spec17 = seq(spec17part1, choose(spec17part2, spec17part5))

    # spec18
    spec18part1 = seq(ev(grid_params.in_room((0, 2))),
                      ev(grid_params.in_room((2, 2))))
    # big loop
    spec18part2 = seq(seq(seq(ev(grid_params.in_room((3, 2))), ev(grid_params.in_room(
        (3, 0)))), ev(grid_params.in_room((2, 0)))), ev(grid_params.in_room((2, 2))))
    spec18part3 = seq(spec18part2, spec18part2)
    for i in range(3):
        spec18part3 = seq(spec18part3, spec18part2)

    # small loop
    spec18part4 = seq(seq(seq(ev(grid_params.in_room((3, 2))), ev(grid_params.in_room(
        (3, 3)))), ev(grid_params.in_room((2, 3)))), ev(grid_params.in_room((2, 2))))
    spec18part5 = seq(spec18part4, spec18part4)
    for i in range(3):
        spec18part5 = seq(spec18part5, spec18part4)

    spec18 = seq(spec18part1, choose(spec18part3, spec18part5))

    # spec19
    spec19part1 = ev(grid_params.in_room((1, 0)))
    # big loop
    spec19part2 = seq(seq(seq(ev(grid_params.in_room((3, 0))), ev(grid_params.in_room(
        (3, 2)))), ev(grid_params.in_room((1, 2)))), ev(grid_params.in_room((1, 0))))

    for i in range(5):
        spec19part1 = seq(spec19part1, spec19part2)

    # small loop
    spec19part3 = seq(ev(grid_params.in_room((0, 2))),
                      ev(grid_params.in_room((2, 2))))
    spec19part4 = seq(seq(seq(ev(grid_params.in_room((3, 2))), ev(grid_params.in_room(
        (3, 3)))), ev(grid_params.in_room((2, 3)))), ev(grid_params.in_room((2, 2))))
    for i in range(5):
        spec19part3 = seq(spec19part3, spec19part4)

    spec19 = choose(spec19part1, spec19part3)
    # spec19 = spec19part1

    specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6,
             spec7, spec8, spec9, spec10, spec11, spec12, spec13, spec14, spec15, spec16, spec17, spec18, spec19]

    for num in spec_nums_set:
        # Step 3: construct abstract reachability graph
        _, abstract_reach = automaton_graph_from_spec(specs[num])
        print('\n**** Abstract Graph ****')
        abstract_reach.pretty_print()

        # get LTL graph
        abstract_LTL_graph = abstract_reach.abstract_graph
        abstract_final_vertices = abstract_reach.final_vertices

        exp_prefix = 'example/%s/rrt_star/' % (env_name,)
        with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
            algo.visualize_evaluate_policy_with_LTL(
                abstract_LTL_graph, abstract_final_vertices, eval_episodes=200, greedy=True, prefix='Eval', env_type=env_name, spec_num=str(num)+'_', searchpolicy=False)


if __name__ == "__main__":
    assert len(sys.argv) == 3
    env_name = str(sys.argv[1])
    spec_num = int(sys.argv[2])
    # assert env_name in ['antumaze', 'antfall', 'antpush', 'antfourrooms']
    params = {
        'seed': [0],
        'env_name': [env_name],
        'gpu': [True],
        'spec_num': [spec_num]
    }
    dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
