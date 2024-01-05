import doodad as dd
import algo.doodad_utils as dd_utils
import sys
import matplotlib.pyplot as plt
import gym
import os 

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

    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, policy, replay_buffer, gcsl_kwargs = variants.get_params(env, env_params)
    graph = Graph((0, 0), (0, 27))

    policy_filename = '/root/code/gcsl_pseudo_algo/ant/data/example/' + env_name + '/demo1/policy.pkl'
    # load trained policy    
    policy.load_state_dict(torch.load(policy_filename))
    policy.eval()

    algo = GCSL_Graph(
        env_name,
        env,
        graph,
        policy,
        None,
        replay_buffer,
        **gcsl_kwargs
    )

    exp_prefix = 'example/%s/' % (env_name,)
    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.evaluate_policy(eval_episodes=10, greedy=True, prefix='Eval', render=True)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    env_name = str(sys.argv[1])
    
    params = {
        'seed': [0],
        'env_name': [env_name],
        'gpu': [True],
    }
    dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
