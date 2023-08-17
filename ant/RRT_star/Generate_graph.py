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
    from gcsl.envs.env_utils import DiscretizedActionEnv

    # Algo
    from gcsl.algo import buffer, gcsl, variants, networks
    from RRT_star.algo import RRTStar_GCSL
    from RRT_star.Graph import Graph
    from RRT_star.v_function_core import value_policy
    import pickle

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    torch.set_num_threads(8)
    np.random.seed(seed)

    env = envs.create_env(env_name)
    env_for_checking = envs.create_env(env_name)
    # env.print_maze_infos()
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, env_for_checking, policy, replay_buffer, gcsl_kwargs = variants.get_params(env, env_for_checking, env_params)
    # print(type(env))
    # print(dir(env))
    RRT_star_tree = Graph((0, 0), (0, 0), algo='Dijkstra')
    valuepolicy = value_policy(env)

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
        algo.grow_RRT_star_tree_using_graph()


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
