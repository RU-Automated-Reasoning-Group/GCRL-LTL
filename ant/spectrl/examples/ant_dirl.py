from spectrl.hierarchy.construction import automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.spec_compiler import ev, seq
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ddpg import DDPGParams
from spectrl.envs.ant import AntEnv
from numpy import linalg as LA

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

num_iters = [1000]


# Define the specification
# 1. Relevant atomic predicates:
# a. Reach predicate
#    goal: np.array(2), err: float
def reach(goal, err):
    def predicate(sys_state, res_state):
        return (-LA.norm(sys_state[:2] - goal) + err)/10
    return predicate


# Goals and obstacles
gtop = np.array([0.0, 5.0])
gbot = np.array([0.0, 0.0])
gright = np.array([5.0, 0.0])
gcorner = np.array([5.0, 5.0])
gcorner2 = np.array([-5.0, 5.0])

# Specifications
spec1 = ev(reach(gtop, 1))
spec2 = seq(ev(reach(gtop, 1)), ev(reach(gbot, 1)))

specs = [spec1, spec2]

lb = [100., 100.]

# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    use_gpu = flags['gpu_flag']
    render = flags['render']
    spec_num = flags['spec_num']
    folder = flags['folder']
    itno = flags['itno']

    log_info = []

    for i in num_iters:

        env = AntEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        hyperparams = DDPGParams(state_dim, action_dim, action_bound,
                                 minibatch_size=100, num_episodes=i,
                                 discount=0.95, actor_hidden_dim=300,
                                 critic_hidden_dim=300, epsilon_decay=1e-6,
                                 decay_function='linear', steps_per_update=1,
                                 buffer_size=200000, sigma=0.2, epsilon_min=0.3)

        print('\n**** Learning Policy for Spec {} ****'.format(spec_num))

        _, abstract_reach = automaton_graph_from_spec(specs[spec_num])
        print('\n**** Abstract Graph ****')
        abstract_reach.pretty_print()

        # Step 5: Learn policy
        abstract_policy, nn_policies, stats = abstract_reach.learn_dijkstra_policy(
            env, hyperparams, max_steps=60, neg_inf=-lb[spec_num], render=render,
            safety_penalty=-1, num_samples=500, algo='ddpg', alpha=0.5, use_gpu=use_gpu)

        # Test policy
        hierarchical_policy = HierarchicalPolicy(
            abstract_policy, nn_policies, abstract_reach.abstract_graph, state_dim)
        final_env = ConstrainedEnv(env, abstract_reach, abstract_policy, max_steps=150)
        _, prob = print_performance(final_env, hierarchical_policy, stateful_policy=True)

        # Print statements
        print('\nTotal Sample Steps: {}'.format(stats[0]))
        print('Total Time Taken: {}'.format(stats[1]))
        print('Total Edges Learned: {}'.format(stats[2]))

        # Render
        if render:
            print('\nSimulation with learned policy...')
            get_rollout(final_env, hierarchical_policy, True, stateful_policy=True)

        logdir = os.path.join(folder, 'spec{}'.format(spec_num), 'hierarchy')
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        log_info.append([stats[0], stats[1], prob])

    save_log_info(log_info, itno, logdir)
