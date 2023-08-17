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

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = envs.create_env(env_name)
    env_for_checking = envs.create_env(env_name)
    env.print_maze_infos()
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, env_for_checking, policy, replay_buffer, gcsl_kwargs = variants.get_params(
        env, env_for_checking, env_params)
    RRT_star_tree = Graph((0, 0), (0, 27))
    policy_filename = '/root/code/gcsl_ant/data/example/' + env_name + '/rrt_star/' + env_name + \
        '_test10_finetune1e6(center)_redo2_SL/policy.pkl'
        # '_test10_finetune1e6(center)_SLcenterSG_3x256_batch512[1000and200]/policy.pkl'
        # '_test6_afterfinetuning8e5randposition+selectclosetondistanceANDcentergoal_SL[800and200]_centralgoal/policy.pkl'
    # load trained policy
    policy.load_state_dict(torch.load(policy_filename))
    policy.eval()

    algo = RRTStar_GCSL(
        env_name,
        env,
        env_for_checking,
        RRT_star_tree,
        policy,
        None,
        replay_buffer,
        **gcsl_kwargs
    )

    exp_prefix = 'example/%s/rrt_star/' % (env_name,)
    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.visualize_evaluate_policy(
            eval_episodes=10, greedy=True, prefix='Eval', env_type=env_name)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    env_name = str(sys.argv[1])
    # assert env_name in  ['antumaze', 'antfall', 'antpush', 'antfourrooms']
    params = {
        'seed': [0],
        'env_name': [env_name],
        'gpu': [True],
    }
    dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
