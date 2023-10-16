import doodad as dd
import algo.doodad_utils as dd_utils
import sys
import matplotlib.pyplot as plt
import gym
from numpy import linalg as LA
import os

def run(output_dir='/tmp', env_name='pointmass_empty', gpu=True, seed=0, **kwargs):

    import gym
    import numpy as np
    from rlutil.logging import log_utils, logger

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu

    # Envs

    from algo import envs

    # Algo
    from algo.gcsl_utils import buffer, variants, networks
    from algo.GCSL_Graph import GCSL_Graph
    from algo.graph_utils.graph import Graph
    from algo.graph_utils.v_function_core import value_policy
    import pickle
    from algo.buchi.scc import path_finding

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)

    # get ant env
    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, policy, replay_buffer, gcsl_kwargs = variants.get_params(
        env, env_params)

    file_path = '/root/code/gcsl_pseudo_algo/ant/data/example/' + env_name + '/ant16rooms_test10_finetune1e6(center)_redo2_SL/'
    policy_filename = file_path + 'policy.pkl'

    # load main policy trained by SL
    policy.load_state_dict(torch.load(policy_filename))
    policy.eval()

    # load value policy and graph
    v_policy = value_policy(env)
    v_policy_filename = file_path + 'value_policy.pkl'
    v_policy.load_policy(v_policy_filename)


    graph = Graph((0, 0), (0, 0), algo='Dijkstra')
    policy = policy.cuda()

    algo = GCSL_Graph(
        env_name,
        env,
        graph,
        policy,
        v_policy,
        replay_buffer,
        **gcsl_kwargs
    )

    def reach(goal, err=1):
        def predicate(sys_state):
            return [err-LA.norm(sys_state[:2]-goal) >= 0, goal]
        return predicate
    
    spec_num = kwargs['spec_num']

    print('\n****  Testing on Spec #{}  ****'.format(spec_num))
    
    room_0_0 = reach(np.array([0,0]))
    room_0_1 = reach(np.array([9,0]))
    room_0_2 = reach(np.array([18,0]))
    room_0_3 = reach(np.array([27,0]))
    

    room_1_0 = reach(np.array([0,9]))
    room_1_1 = reach(np.array([9,9]))
    room_1_2 = reach(np.array([18,9]))
    room_1_3 = reach(np.array([27,9]))

    room_2_0 = reach(np.array([0,18]))
    room_2_2 = reach(np.array([18,18]))
    room_2_1 = reach(np.array([9,18]))
    room_2_3 = reach(np.array([27,18]))

    room_3_0 = reach(np.array([0,27]))
    room_3_1 = reach(np.array([9,27]))
    room_3_2 = reach(np.array([18,27]))
    room_3_3 = reach(np.array([27,27]))

    predicates = {
        'empty': room_0_0,
        'room_0_0': room_0_0,
        'room_0_1': room_0_1,
        'room_0_2': room_0_2,
        'room_0_3': room_0_3,
        'room_1_0': room_1_0,
        'room_1_1': room_1_1,
        'room_1_2': room_1_2,
        'room_1_3': room_1_3,
        'room_2_0': room_2_0,
        'room_2_1': room_2_1,
        'room_2_2': room_2_2,
        'room_2_3': room_2_3,
        'room_3_0': room_3_0,
        'room_3_1': room_3_1,
        'room_3_2': room_3_2,
        'room_3_3': room_3_3,
    }

    # test specs
    spec0 = 'F( room_0_1 ) '  

    # \phi1
    spec1 = 'F( room_0_2 || room_2_0 )'

    # \phi2
    spec2 = 'F( (room_0_2 || room_2_0) && F( room_2_2 ))'

    # \phi3
    spec3 = 'F( (room_0_2 || room_2_0) && F( room_2_2 && F( (room_2_1 || room_3_2) && F( room_3_1 ))))'

    # \phi4
    spec4 = 'F( (room_0_2 || room_2_0) && F( room_2_2 && F( (room_2_1 || room_3_2) && F( room_3_1 && F(( room_1_1 || room_3_3 ) && F( room_1_3 ))))))'
    
    # \phi5
    spec5 = 'F( (room_0_2 || room_2_0) && F( room_2_2 && F( (room_2_1 || room_3_2) && F( room_3_1 && F(( room_1_1 || room_3_3 ) && F( room_1_3 && F(( room_1_1 || room_0_3 ) && F( room_0_1 ))))))))'

    # \phi omega
    spec6 = 'GF( room_1_0 && XF( room_3_0 && XF(room_3_2 && XF(room_1_2)))) || (F room_0_2 && XGF( room_2_2 && XF( room_3_2 && XF( room_3_3 && XF( room_2_3)))))'

    specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6]
    goals,avoids = path_finding(specs[spec_num], v_policy, predicates, debug=True)
    goals = [predicates[goal] for goal in goals]

    exp_prefix = 'evaluate/%s/' % (env_name,)
    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.visualize_evaluate_policy_with_LTL_buchi(goals, avoids, eval_episodes=10, greedy=True, prefix='Eval', env_type=env_name, spec_num=str(spec_num)+'_')


if __name__ == "__main__":
    assert len(sys.argv) == 3
    env_name = str(sys.argv[1])
    spec_num = int(sys.argv[2])
    params = {
        'seed': [0],
        'env_name': [env_name],
        'gpu': [True],
        'iter': [iter],
        'spec_num': [spec_num],
    }
    dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
