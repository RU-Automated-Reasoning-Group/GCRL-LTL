from spectrl.hierarchy.construction import automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.monitor import Resource_Model
from spectrl.main.spec_compiler import ev, alw, seq, choose
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ars import HyperParams
from spectrl.envs.car2d import VC_Env
from numpy import linalg as LA

import numpy as np
import os

num_iters = [50, 100, 200, 300]


# Define the resource model
# Fuel consumption proportional to distance from x-axis and the velocity
# sys_state: np.array(2)
# res_state: np.array(1)
# sys_action: np.array(2)
def resource_delta(sys_state, res_state, sys_action):
    return np.array([res_state[0] - 0.1 * abs(sys_state[0]) * LA.norm(sys_action)])


# Define the specification
# 1. Relevant atomic predicates:
# a. Reach predicate
#    goal: np.array(2), err: float
def reach(goal, err):
    def predicate(sys_state, res_state):
        return (min([sys_state[0] - goal[0],
                     goal[0] - sys_state[0],
                     sys_state[1] - goal[1],
                     goal[1] - sys_state[1]]) + err)
    return predicate


# b. Avoid predicate
#    obstacle: np.array(4): [x_min, y_min, x_max, y_max]
def avoid(obstacle):
    def predicate(sys_state, res_state):
        return 10 * max([obstacle[0] - sys_state[0],
                         obstacle[1] - sys_state[1],
                         sys_state[0] - obstacle[2],
                         sys_state[1] - obstacle[3]])
    return predicate


def have_fuel(sys_state, res_state):
    return res_state[0]


# Goals and obstacles
gtop = np.array([5.0, 10.0])
gbot = np.array([5.0, 0.0])
gright = np.array([10.0, 0.0])
gcorner = np.array([10.0, 10.0])
gcorner2 = np.array([0.0, 10.0])
origin = np.array([0.0, 0.0])
obs = np.array([4.0, 4.0, 6.0, 6.0])

# Specifications
spec1 = alw(avoid(obs), ev(reach(gtop, 1.0)))
spec2 = alw(avoid(obs), alw(have_fuel, ev(reach(gtop, 1.0))))
spec3 = seq(alw(avoid(obs), ev(reach(gtop, 1.0))),
            alw(avoid(obs), ev(reach(gbot, 1.0))))
spec4 = seq(choose(alw(avoid(obs), ev(reach(gtop, 1.0))), alw(avoid(obs), ev(reach(gright, 1.0)))),
            alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec5 = seq(spec3, alw(avoid(obs), ev(reach(gright, 1.0))))
spec6 = seq(spec5, alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec7 = seq(spec6, alw(avoid(obs), ev(reach(origin, 1.0))))


# Examples: Choice but greedy doesn't work
gt1 = np.array([3.0, 4.0])
gt2 = np.array([6.0, 0.0])
gfinal = np.array([3.0, 7.0])
gfinal2 = np.array([7.0, 4.0])

spec8 = seq(choose(alw(avoid(obs), ev(reach(gt2, 0.5))), alw(avoid(obs), ev(reach(gt1, 0.5)))),
            alw(avoid(obs), ev(reach(gfinal, 0.5))))

spec9 = seq(choose(alw(avoid(obs), ev(reach(gt2, 0.5))), alw(avoid(obs), ev(reach(gt1, 0.5)))),
            alw(avoid(obs), ev(reach(gfinal2, 0.5))))


spec10 = choose(alw(avoid(obs), ev(reach(gt1, 0.5))),
                alw(avoid(obs), ev(reach(gt2, 0.5))))

specs = [spec1, spec2, spec3, spec4, spec5, spec6, spec7, spec8, spec9, spec10]

lb = [10., 20., 10., 10., 10., 9., 9., 9., 9., 9.]

# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    render = flags['render']
    folder = flags['folder']
    itno = flags['itno']
    spec_num = flags['spec_num']

    log_info = []

    for i in num_iters:
        hyperparams = HyperParams(30, i, 20, 8, 0.05, 1, 0.2)

        print('\n**** Learning Policy for Spec {} for {} Iterations ****'.format(spec_num, i))

        # Step 1: initialize system environment
        system = VC_Env(500, std=0.05)

        # Step 2 (optional): construct resource model
        resource = Resource_Model(2, 2, 1, np.array([7.0]), resource_delta)

        # Step 3: construct abstract reachability graph
        _, abstract_reach = automaton_graph_from_spec(specs[spec_num])
        print('\n**** Abstract Graph ****')
        abstract_reach.pretty_print()

        # Step 5: Learn policy
        abstract_policy, nn_policies, stats = abstract_reach.learn_dijkstra_policy(
            system, hyperparams, res_model=resource, max_steps=20,
            neg_inf=-lb[spec_num], safety_penalty=-1, num_samples=500, render=render)

        # Test policy
        hierarchical_policy = HierarchicalPolicy(
            abstract_policy, nn_policies, abstract_reach.abstract_graph, 2)
        final_env = ConstrainedEnv(system, abstract_reach, abstract_policy,
                                   res_model=resource, max_steps=60)

        # Print statements
        _, prob = print_performance(final_env, hierarchical_policy, stateful_policy=True)
        print('\nTotal Sample Steps: {}'.format(stats[0]))
        print('Total Time Taken: {} mins'.format(stats[1]))
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
