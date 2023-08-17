from spectrl.hierarchy.construction import automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.spec_compiler import ev, seq, choose
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout, ObservationWrapper
from spectrl.rl.ddpg import DDPGParams
from spectrl.envs.fetch import FetchPickAndPlaceEnv
from numpy import linalg as LA

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

num_iters = [1000, 2000, 4000]


def grip_near_object(err):
    def predicate(sys_state, res_state):
        dist = sys_state[:3] - (sys_state[3:6] + np.array([0., 0., 0.065]))
        dist = np.concatenate([dist, [sys_state[9] + sys_state[10] - 0.1]])
        return -LA.norm(dist) + err
    return predicate


def hold_object(err):
    def predicate(sys_state, res_state):
        dist = sys_state[:3] - sys_state[3:6]
        dist2 = np.concatenate([dist, [sys_state[9] + sys_state[10] - 0.045]])
        return -LA.norm(dist2) + err
    return predicate


def object_in_air(sys_state, res_state):
    return sys_state[5] - 0.45


def object_at_goal(err):
    def predicate(sys_state, res_state):
        dist = np.concatenate([sys_state[-3:], [sys_state[9] + sys_state[10] - 0.045]])
        return -LA.norm(dist) + err
    return predicate


def gripper_reach(goal, err):
    '''
    goal: numpy array of dim (3,)
    '''
    def predicate(sys_state, res_state):
        return -LA.norm(sys_state[:3] - goal) + err
    return predicate


def object_reach(goal, err):
    '''
    goal: numpy array of dim (3,)
    '''
    def predicate(sys_state, res_state):
        return -LA.norm(sys_state[3:6] - goal) + err
    return predicate


above_corner1 = np.array([1.15, 1.0, 0.5])
above_corner2 = np.array([1.45, 1.0, 0.5])
corner1 = np.array([1.15, 1.0, 0.425])
corner2 = np.array([1.50, 1.05, 0.425])

# Specifications
spec1 = ev(grip_near_object(0.03))
spec2 = seq(spec1, ev(hold_object(0.03)))
spec3 = seq(spec2, ev(object_at_goal(0.05)))
spec4 = seq(seq(spec2, ev(object_in_air)), ev(object_at_goal(0.05)))
spec5 = seq(seq(spec2, ev(object_in_air)), ev(object_reach(above_corner1, 0.05)))
spec6 = seq(seq(spec2, ev(object_in_air)),
            choose(seq(ev(object_reach(above_corner1, 0.05)), ev(object_reach(corner1, 0.05))),
                   seq(ev(object_reach(above_corner2, 0.05)), ev(object_reach(corner2, 0.01)))))

specs = [spec1, spec2, spec3, spec4, spec5, spec6]

lb = [100., 100., 100., 100., 100., 100.]

# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    use_gpu = flags['gpu_flag']
    render = flags['render']
    folder = flags['folder']
    itno = flags['itno']
    spec_num = flags['spec_num']

    log_info = []

    for i in num_iters:

        env = ObservationWrapper(FetchPickAndPlaceEnv(), ['observation', 'desired_goal'],
                                 relative=(('desired_goal', 0, 3), ('observation', 3, 6)))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        hyperparams = DDPGParams(state_dim, action_dim, action_bound,
                                 minibatch_size=256, num_episodes=i,
                                 discount=0.95, actor_hidden_dim=256,
                                 critic_hidden_dim=256, epsilon_decay=3e-6,
                                 decay_function='linear', steps_per_update=100,
                                 gradients_per_update=100, buffer_size=200000,
                                 sigma=0.15, epsilon_min=0.3, target_noise=0.0003,
                                 target_clip=0.003, warmup=1000)

        print('\n**** Learning Policy for Spec {} ****'.format(spec_num))

        _, abstract_reach = automaton_graph_from_spec(specs[spec_num])
        print('\n**** Abstract Graph ****')
        abstract_reach.pretty_print()

        # Step 5: Learn policy
        abstract_policy, nn_policies, stats = abstract_reach.learn_dijkstra_policy(
            env, hyperparams, max_steps=40, neg_inf=-lb[spec_num], render=render,
            safety_penalty=-1, num_samples=1000, algo='ddpg', alpha=0, use_gpu=use_gpu)

        # Test policy
        hierarchical_policy = HierarchicalPolicy(
            abstract_policy, nn_policies, abstract_reach.abstract_graph, state_dim)
        final_env = ConstrainedEnv(env, abstract_reach, abstract_policy, max_steps=200)
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
