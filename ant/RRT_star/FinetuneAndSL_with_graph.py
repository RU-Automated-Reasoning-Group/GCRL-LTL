import doodad as dd
import gcsl.doodad_utils as dd_utils
import sys
import matplotlib.pyplot as plt
import gym
import os

def run(output_dir='/tmp', env_name='pointmass_empty', gpu=True, seed=0, samples_num='2000', training_time=200, **kwargs):

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
    # from RRT_star import buffer

   


    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.set_num_threads(8)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = envs.create_env(env_name)
    env_for_checking = envs.create_env(env_name)
    env.print_maze_infos()
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, env_for_checking, policy, replay_buffer, gcsl_kwargs = variants.get_params(env, env_for_checking, env_params)
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
        algo.train_SL_using_graph(samples_num, training_time)


if __name__ == "__main__":
    assert len(sys.argv) == 5
    env_name = str(sys.argv[1])
    samples_num = int(sys.argv[2])
    training_time = int(sys.argv[3])
    seed = int(sys.argv[4])
    # assert env_name in ['antumaze', 'antfall', 'antpush', 'antfourrooms']
    params = {
        'seed': [seed],
        'env_name': [env_name],
        'gpu': [True],
        'samples_num': [samples_num],
        'training_time': [training_time],
    }
    dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
