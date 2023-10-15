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
    from algo.graph_utils.graph import Graph
    from algo.graph_utils.v_function_core import value_policy
    import pickle

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

    env, policy, replay_buffer, gcsl_kwargs = variants.get_params(env, env_params)
    graph = Graph((0, 0), (0, 0), algo='Dijkstra')
    valuepolicy = value_policy(env)

    filepath = '/root/code/gcsl_ant/data/example/' + env_name + '/rrt_star/' + env_name + \
        '_test10/'
    filename = filepath + 'value_policy.pkl'
    # valuepolicy.load_policy(filename)

    graph_filename = filepath + 'graph.pkl'
    with open(graph_filename, 'rb') as f:
        graph = pickle.load(f)

    graph.set_algo('Dijkstra')
    policy = policy.cuda()

    algo = GCSL_Graph(
        env_name,
        env,
        graph,
        policy,
        valuepolicy,
        replay_buffer,
        **gcsl_kwargs
    )

    exp_prefix = 'example/%s/rrt_star/' % (env_name,)
    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.finetune_value_Vpolicy()


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
