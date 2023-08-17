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

    import envs
    from envs.env_utils import DiscretizedActionEnv

    # Algo
    from algo import buffer, variants, networks
    from algo.algo import RRTStar_GCSL
    from algo.Graph import Graph
    from algo.v_function_core import value_policy
    import pickle
    # from RRT_star import buffer

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)

    env = envs.create_env(env_name)
    env_for_checking = envs.create_env(env_name)
    env.print_maze_infos()
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, env_for_checking, policy, replay_buffer, gcsl_kwargs = variants.get_params(env, env_for_checking, env_params)
    RRT_star_tree = Graph((0, 0), (0, 0), algo='Dijkstra')
    valuepolicy = value_policy(env)

    filepath = '/root/code/gcsl_ant/data/example/' + env_name + '/rrt_star/' + env_name + \
        '_test10/'
    filename = filepath + 'value_policy.pkl'
    # valuepolicy.load_policy(filename)

    tree_filename = filepath + 'RRT_star_tree.pkl'
    #RRT_star_tree = Tree((0, 0), (0, 27))
    with open(tree_filename, 'rb') as f:
        RRT_star_tree = pickle.load(f)

    RRT_star_tree.set_algo('Dijkstra')
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

    exp_prefix = 'example/%s/rrt_star/' % (env_name,)
    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.fine_tuning_value_policy()


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
