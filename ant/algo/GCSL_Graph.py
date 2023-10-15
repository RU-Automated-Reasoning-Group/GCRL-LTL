import numpy as np
from torch import threshold
from rlutil.logging import logger

import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu
import math
import time
import tqdm
import os.path as osp
import copy
import pickle
import matplotlib.pyplot as plt
from envs.env_utils import DiscretizedActionEnv
import warnings

from algo.graph_utils.utils import *
from algo.graph_utils.v_function_core import *
try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_enabled = True
except:
    print('Tensorboard not installed!')
    tensorboard_enabled = False


class GCSL_Graph:
    def __init__(self,
                 env_name,
                 env,
                 env_for_checking,
                 graph,
                 policy,
                 value_policy,
                 replay_buffer,
                 validation_buffer,
                 rrt_tree_node_cover_size=1,
                 max_timesteps=1e6,
                 max_path_length=50,
                 # Exploration Strategy
                 explore_timesteps=1e4,
                 expl_noise=0.1,
                 # Evaluation / Logging
                 goal_threshold=0.05,
                 eval_freq=5e3,
                 eval_episodes=200,
                 save_every_iteration=False,
                 log_tensorboard=False,
                 # Policy Optimization Parameters
                 start_policy_timesteps=0,
                 batch_size=100,
                 n_accumulations=1,
                 policy_updates_per_step=1,
                 train_policy_freq=None,
                 lr=5e-4
                 ):
        self.env_name = env_name
        self.env = env
        self.env_for_checking = env_for_checking
        self.graph = graph

        self.policy = policy
        self.value_policy = value_policy
        self.replay_buffer = replay_buffer
        self.validation_buffer = validation_buffer

        self.rrt_tree_node_cover_size = rrt_tree_node_cover_size

        self.is_discrete_action = hasattr(self.env.action_space, 'n')

        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length

        self.explore_timesteps = explore_timesteps
        self.expl_noise = expl_noise

        self.goal_threshold = goal_threshold
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_every_iteration = save_every_iteration

        self.start_policy_timesteps = start_policy_timesteps
        if train_policy_freq is None:
            train_policy_freq = self.max_path_length

        self.train_policy_freq = train_policy_freq
        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr)

        self.log_tensorboard = log_tensorboard and tensorboard_enabled
        self.summary_writer = None




    def sample_trajectory_for_evaluation(self, greedy=False, noise=0):
        state = self.env.reset()

        if self.env_name in ['ant16rooms']:

            rooms = [[0, 0], [0, 18], [0, 27],
                     [9, 0], [9, 9], [9, 18], [9, 27],
                     [18, 0], [18, 9], [18, 18], [18, 27],
                     [27, 0], [27, 9],  [27, 27]]

            # for ant16rooms
            start_pos = random.choice(rooms)
            random_start_pos = start_pos + \
                np.random.uniform(size=2, low=-1, high=1)
            state = self.env.reset_with_init_pos_at(
                random_start_pos)

            # try only picking goal at the center range of rooms
            goal_state = self.env.sample_goal()

            goal = random.choice(rooms)
            while goal == start_pos:
                goal = random.choice(rooms)
            goal = goal + np.random.uniform(size=2, low=-1, high=1)

            goal_state[:2] = goal

        # keep tracking of trajectory
        states = []
        actions = []

        for t in range(self.max_path_length):

            states.append(state)
            # get observation
            observation = self.env.observation(state)
            horizon = np.arange(self.max_path_length) >= (
                self.max_path_length - 1 - t)  # Temperature encoding of horizon
            action = self.policy.act_vectorized(
                observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
            if not self.is_discrete_action:
                action = np.clip(action, self.env.action_space.low,
                                 self.env.action_space.high)

            actions.append(action)
            # execute action on current state
            state, _, done, _ = self.env.step(action)

            if done == False:
                done = True if np.linalg.norm(
                    (states[-1][:2] - goal), axis=-1) < self.goal_threshold else False

            # if done == True:
            #     # states.append(state)
            #     break
        trajectory_length = len(states)

        return np.stack(states), np.array(actions), goal_state, trajectory_length

    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval', total_timesteps=0, render=False):
        env = self.env

        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        avg_length = 0
        max_length = 0
        succeeded_count = 0
        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state, trajectory_len = self.sample_trajectory_for_evaluation(noise=0, greedy=greedy)
            assert len(states) == len(actions)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(
                states[trajectory_len-1], goal_state)

            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        logger.record_tabular('%s num episodes' % prefix, eval_episodes)
        logger.record_tabular('%s avg final dist' %
                              prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio' %
                              prefix, np.mean(success_vec))
        if self.summary_writer:
            self.summary_writer.add_scalar(
                '%s/avg final dist' % prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar(
                '%s/success ratio' % prefix,  np.mean(success_vec), total_timesteps)
        if render:
            diagnostics = env.get_diagnostics(
            all_states, all_goal_states, success_vec)
            for key, value in diagnostics.items():
                logger.record_tabular('%s %s' % (prefix, key), value)
        return all_states, all_goal_states

    def visualize_evaluate_policy_with_LTL(self, abstract_LTL_graph, abstract_final_vertices, eval_episodes, greedy=True, 
                                           prefix='Eval', total_timesteps=0, env_type="none",spec_num=""):
        env = self.env
        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state, trajectory_len = self.sample_trajectory_for_evaluation_with_LTL(
                abstract_LTL_graph, abstract_final_vertices, noise=0, greedy=greedy, goal_env=env_type)

            assert len(states) == len(actions)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)

            final_dist = env.goal_distance(
                states[trajectory_len-1], goal_state)
            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        logger.record_tabular('%s num episodes' % prefix, eval_episodes)
        logger.record_tabular('%s avg final dist' %
                              prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio' %
                              prefix, np.mean(success_vec))
        if self.summary_writer:
            self.summary_writer.add_scalar(
                '%s/avg final dist' % prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar(
                '%s/success ratio' % prefix,  np.mean(success_vec), total_timesteps)

        diagnostics = env.get_diagnostics(
            all_states, all_goal_states, success_vec, spec_num)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s' % (prefix, key), value)

        return all_states, all_goal_states

    def sample_trajectory_for_evaluation_with_LTL(self, abstract_LTL_graph, abstract_final_vertices, greedy=True, noise=0, render=False):
        # get sequence of tasks that having smallest accumulated '1-v' vlaue (dijkstra algo)
        num_nodes = len(abstract_LTL_graph)

        target_map = {0: (0, 0)}
        neighbors = {}
        for i in range(num_nodes):
            edges_from_curnode = abstract_LTL_graph[i]
            neighbors[i] = []
            for edge in edges_from_curnode:
                neighbors[i].append(edge.target)
                if i != edge.target and target_map.get(edge.target) == None:
                    # adjust the form of room to fit ant env
                    target_map[edge.target] = [edge.predicate(
                        (0, 0), [])[0][1], edge.predicate((0, 0), [])[0][0]]

        nodes = [i for i in range(num_nodes)]
        distance = []
        predecessor = []
        # initialize graph
        for node in nodes:
            distance.append(float('inf'))
            predecessor.append(-1)
        # dist=0 for the start node
        distance[0] = 0

        while nodes:
            curNode = min(nodes, key=lambda node: distance[node])
            nodes.remove(curNode)
            if distance[curNode] == float('inf'):
                break

            for to_node in neighbors[curNode]:

                cur_state = [
                    x * self.env.room_scaling for x in target_map[curNode]]
                to_state = [
                    x * self.env.room_scaling for x in target_map[to_node]]
                value = self.value_policy.get_value(
                    cur_state, to_state)
                
                weight = 1 - value
                if weight < 0:
                    weight = 0

                # if value <= 0:
                #     value = 1e-6
                # elif value >= 1:
                #     value = 1 - 1e-6
                # weight = -math.log(value)


                newCost = distance[curNode] + weight
                if newCost < distance[to_node]:
                    distance[to_node] = newCost
                    predecessor[to_node] = curNode
        final_goal_vertex = min(abstract_final_vertices,
                                key=lambda vertex: distance[vertex])

        subtasks_queue = deque()
        cur_vertex = final_goal_vertex
        while predecessor[cur_vertex] != -1:
            subtasks_queue.appendleft(cur_vertex)
            cur_vertex = predecessor[cur_vertex]

        subgoals = deque()
        for subtask in subtasks_queue:
            subgoals.append(
                np.array(target_map[subtask]) * self.env.room_scaling)

        max_trajectory_len = len(subtasks_queue)*self.max_path_length
        # keep tracking of trajectory
        states_of_alltasks = []
        actions_of_alltasks = []

        goal_state = self.env.sample_goal()
        state = self.env.reset()

        while len(subtasks_queue) != 0:
            sub_goal = np.array(target_map[subtasks_queue.popleft(
            )]) * self.env.room_scaling
            goal_state[:2] = sub_goal
            done = False
            states = []
            actions = []
            for t in range(self.max_path_length):
                if done == True:
                    break
                # get observation
                observation = self.env.observation(state)

                horizon = np.arange(self.max_path_length) >= (
                    self.max_path_length - 1 - t)  # Temperature encoding of horizon
                action = self.policy.act_vectorized(
                    observation[None], sub_goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
                if not self.is_discrete_action:
                    action = np.clip(
                        action, self.env.action_space.low, self.env.action_space.high)

                actions.append(action)
                # execute action on current state
                state, _, _, _ = self.env.step(action)
                states.append(state)

                if done == False:
                    done = True if np.linalg.norm(
                        (state[:2] - sub_goal), axis=-1) < self.goal_threshold else False

            states_of_alltasks += states
            actions_of_alltasks += actions
            # current subtask isn't solved
            if done == False:
                break

        trajectory_length = len(states_of_alltasks)

        # pad the states and actions to be the same size (max_path_length * state.shape/action.shape)
        if trajectory_length < max_trajectory_len:
            pad_len = max_trajectory_len - trajectory_length
            pad_state = states_of_alltasks[-1]
            pad_action = 0
            for i in range(pad_len):
                states_of_alltasks.append(pad_state)
                actions_of_alltasks.append(pad_action)

        return np.stack(states_of_alltasks), np.array(actions_of_alltasks), goal_state, trajectory_length



    def generate_graph(self):
        start_time = time.time()
        last_time = start_time

        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0

        if logger.get_snapshot_dir() and self.log_tensorboard:
            self.summary_writer = SummaryWriter(
                osp.join(logger.get_snapshot_dir(), 'tensorboard'))

        ###
        sample_goals_x = []
        sample_goals_y = []
        ###
        fig_loss_x = []
        fig_loss_y = []
        running_loss = None
        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:

                # Interact in environmenta according to exploration strategy.
                states, _, goal_state, _, opt_path, dataset, penalty = self.sample_trajectory()
                loss = self.value_policy.execute(dataset, penalty, self.rrt_tree_node_cover_size)
                
                sample_goals_x.append(goal_state[0])
                sample_goals_y.append(goal_state[1])

                fig_loss_x.append(total_timesteps/self.max_path_length)
                fig_loss_y.append(loss.detach().numpy())

                total_timesteps += self.max_path_length
                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length

                ranger.update(self.max_path_length)

                # Evaluate, log, and save to disk
                if timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    logger.record_tabular('timesteps', total_timesteps)
                    logger.record_tabular(
                        'epoch time (s)', time.time() - last_time)
                    logger.record_tabular(
                        'total time (s)', time.time() - start_time)
                    last_time = time.time()
                    logger.dump_tabular()
                    ranger.reset()
                    self.save_rrt_star_tree(
                        total_timesteps, sample_goals_x, sample_goals_y, 10, opt_path)
        self.save_rrt_star_tree(
            total_timesteps, sample_goals_x, sample_goals_y, 1, opt_path, states)
        self.save_loss_fig("value_policy_loss", fig_loss_x, fig_loss_y)
        self.env.sample_goal_scatter_fig(sample_goals_x, sample_goals_y)
    
    
    def finetune_value_Vpolicy(self):
        start_time = time.time()
        last_time = start_time

        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0

        if logger.get_snapshot_dir() and self.log_tensorboard:
            self.summary_writer = SummaryWriter(
                osp.join(logger.get_snapshot_dir(), 'tensorboard'))

        ###
        sample_goals_x = []
        sample_goals_y = []
        ###

        total_trace = 0
        succeeded_trace = 0
        fig_loss_x = []
        fig_loss_y = []
        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < int(self.max_timesteps):
                # Interact in environmenta according to exploration strategy.
                states, _, goal_state, _, opt_path, dataset, penalty = self.sample_trajectory(finetune=True)
                loss = self.value_policy.execute(
                    dataset, penalty, self.rrt_tree_node_cover_size)

                fig_loss_x.append(total_timesteps/self.max_path_length)
                fig_loss_y.append(loss.detach().numpy())

                total_trace += 1
                if not penalty:
                    succeeded_trace += 1
                ###
                sample_goals_x.append(goal_state[0])
                sample_goals_y.append(goal_state[1])
                ###

                total_timesteps += self.max_path_length
                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length

                ranger.update(self.max_path_length)

                # Evaluate, log, and save to disk
                if timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    logger.record_tabular('timesteps', total_timesteps)
                    logger.record_tabular(
                        'epoch time (s)', time.time() - last_time)
                    logger.record_tabular(
                        'total time (s)', time.time() - start_time)
                    last_time = time.time()
                    logger.dump_tabular()
                    ranger.reset()
                    self.save_rrt_star_tree(
                        total_timesteps, sample_goals_x, sample_goals_y, 10, opt_path, states)

            self.save_loss_fig("value_policy_loss", fig_loss_x, fig_loss_y)

    def supervised_learning(self, samples_num, training_times):
        # Evaluate untrained policy
        total_timesteps = 0

        running_loss = None
        running_validation_loss = None

        if logger.get_snapshot_dir() and self.log_tensorboard:
            self.summary_writer = SummaryWriter(
                osp.join(logger.get_snapshot_dir(), 'tensorboard'))

        ###
        fig_loss_x = []
        fig_loss_y = []
        fig_validation_loss_y = []

        total_trace = 0
        succeeded_trace = 0
        with tqdm.tqdm(total=samples_num, smoothing=0) as ranger:
            for n in range(samples_num):
                states, actions, goal_state, trajectory_len, opt_path, done = self.sample_trajectory_for_SL()

                if done:

                    assert len(states) == len(actions)
                    self.replay_buffer.add_trajectory(
                        states, actions, goal_state, length_of_traj=trajectory_len)
                    # if self.validation_buffer is not None and np.random.rand() < 0.2:
                    #     self.validation_buffer.add_trajectory(
                    #         states, actions, goal_state, length_of_traj=trajectory_len)
                ranger.update(1)


        with tqdm.tqdm(total=training_times, smoothing=0) as ranger:
            start_out_of_rollout_training_time = time.time()
            for x in range(int(training_times)):
                running_loss = None
                running_validation_loss = None
                for _ in range(int(self.train_policy_freq)):
                    self.policy.train()
                    loss = self.take_policy_step()
                    validation_loss = self.validation_loss()
                    if running_loss is None:
                        running_loss = loss
                    else:
                        running_loss += loss

                    if running_validation_loss is None:
                        running_validation_loss = validation_loss
                    else:
                        running_validation_loss += validation_loss

                running_loss = running_loss/self.train_policy_freq
                running_validation_loss = running_validation_loss/self.train_policy_freq
                fig_loss_x.append(x)
                fig_loss_y.append(running_loss)
                fig_validation_loss_y.append(running_validation_loss)
                ranger.update(1)

                self.policy.eval()
                ranger.set_description('Loss: %s Validation Loss: %s' % (
                    running_loss, running_validation_loss))

                if self.summary_writer:
                    self.summary_writer.add_scalar(
                        'Losses/Train', running_loss, total_timesteps)
                    self.summary_writer.add_scalar(
                        'Losses/Validation', running_validation_loss, total_timesteps)

                if (x + 1) % 50 == 0:
                    self.evaluate_policy(eval_episodes=20, greedy=True, prefix='Eval', render=False)
                    if logger.get_snapshot_dir():
                        modifier = str(x)
                        torch.save(
                            self.policy.state_dict(),
                            osp.join(logger.get_snapshot_dir(),'policy%s.pkl' % modifier)
                        )

                        full_dict = dict(env=self.env, policy=self.policy)
                        with open(osp.join(logger.get_snapshot_dir(), 'params%s.pkl' % modifier), 'wb') as f:
                            pickle.dump(full_dict, f)
                    self.save_loss_fig("SL_graph" + modifier, fig_loss_x,fig_loss_y, validation_loss=fig_validation_loss_y, )

        self.save_loss_fig("SL_graph", fig_loss_x, fig_loss_y,validation_loss=fig_validation_loss_y)


    def sample_trajectory(self, finetune=False):

        states = []
        actions = []
        if finetune:
            rooms = [[0, 0], [0, 18], [0, 27],
                    [9, 0], [9, 9], [9, 18], [9, 27],
                    [18, 0], [18, 9], [18, 18], [18, 27],
                    [27, 0], [27, 9],  [27, 27]]

            start_pos = random.choice(rooms)
            random_start_pos = start_pos + \
                np.random.uniform(size=2, low=-1, high=1)
            current_state = self.env.reset_with_init_pos_at(
                random_start_pos)

            # try only picking goal at the center of rooms
            goal_state = self.env.sample_goal()

            goal = random.choice(rooms)
            while goal == start_pos:
                goal = random.choice(rooms)
            goal = goal + np.random.uniform(size=2, low=-1, high=1)
            goal_state[:2] = goal
        else:
            current_state = self.env.reset()
            goal_state = self.env.sample_goal()
            goal = self.env.extract_goal(goal_state)
        
        previous_state = current_state
        current_node = self.get_node_by_state(current_state)
        current_node_idx = self.graph.vex2idx[current_node]        
        
        states = []
        actions = []
        dataset = []

        if finetune:
            opt_path = self.graph.find_path_to_closest_node(current_node_idx, goal, self.value_policy, closest_node_mode="by_distance")
        else:
            opt_path = self.graph.find_path_to_closest_node(current_node_idx, goal, self.value_policy, closest_node_mode="by_value")
        
        remaining_path = opt_path[:]
        endnode_of_optpath = self.graph.vertices[opt_path[0]]
        reached_endnode_of_optpath = False
        done = False
        extended = False

        for t in range(self.max_path_length):

            if done == True:
                extended = True
                break

            high_level_actionidx = 0
            current_node = self.get_node_by_state(current_state)
            prev_node = self.get_node_by_state(previous_state)
            prev_nodeidx = current_node_idx
            try:
                current_node_idx = self.graph.vex2idx[current_node]
            except:
                current_node_idx = self.graph.add_vex(current_node)
                extended = True

            dist = distance(current_node, prev_node)
            if prev_nodeidx != current_node_idx:
                self.graph.add_edge(current_node_idx, prev_nodeidx, dist)
                
            if endnode_of_optpath == current_node:
                reached_endnode_of_optpath = True
            
            if reached_endnode_of_optpath == False:
                dist_to_pathnodes = []
                for node_idx in remaining_path:
                    dist_to_pathnodes.append(
                        (node_idx, distance(self.graph.vertices[node_idx], current_node)))
                dist_to_pathnodes.sort(key=lambda x: x[1])
                closest_idx, closest_dist = dist_to_pathnodes[0]

                if closest_dist != 0:
                    # node is off path
                    target_node_idx, target_node_dist = dist_to_pathnodes[0]
                    high_level_action, high_level_actionidx = self.assign_highlevel_action(
                        current_node, self.graph.vertices[target_node_idx])
                else:
                    ###
                    # if current state is contained with in a node and the node has children which is closer to goal,
                    # take the action on the first edge of the path
                    ###
                    idx_path = remaining_path.index(closest_idx)
                    remaining_path = remaining_path[:idx_path]
                    high_level_action, high_level_actionidx = self.assign_highlevel_action(
                        current_node, self.graph.vertices[remaining_path[-1]])
            else:
                high_level_action, high_level_actionidx = self.assign_highlevel_action(
                    current_node, goal)

            states.append(current_state)
            previous_state = current_state

            actions.append(high_level_actionidx)

            # execute action on current state
            current_state, reward, _, info = self.env.step(
                high_level_actionidx)

            if done == False:
                done = True if np.linalg.norm(
                    (states[-1][:2] - goal), axis=-1) < self.goal_threshold else False

            reward = 0
            data = {}
            data['obs'] = previous_state[:2]
            data['act'] = high_level_action
            data['goal'] = goal
            data['rew'] = reward
            data['obs2'] = current_state[:2]
            data['done'] = done
            dataset.append(data)

        if done == True:
            dataset[-1]['rew'] = 1

        penalty = False
        if done == False and extended == False:
            penalty = True

        trajectory_length = len(states)

        return np.stack(states), np.array(actions), goal_state, trajectory_length, opt_path, dataset, penalty
    
    def sample_trajectory_for_SL(self):
        states = []
        actions = []

        rooms = [[0, 0], [0, 18], [0, 27],
                 [9, 0], [9, 9], [9, 18], [9, 27],
                 [18, 0], [18, 9], [18, 18], [18, 27],
                 [27, 0], [27, 9],  [27, 27]]

        start_pos = random.choice(rooms)
        random_start_pos = start_pos + \
            np.random.uniform(size=2, low=-1, high=1)
        current_state = self.env.reset_with_init_pos_at(random_start_pos)
        previous_state = current_state

        init_pos = self.get_node_by_state(current_state)
        init_pos_idx = self.graph.vex2idx[init_pos]

        endnode_of_optpath = init_pos

        # try only picking goal at the center of rooms
        goal_state = self.env.sample_goal()

        goal = random.choice(rooms)
        while goal == start_pos:
            goal = random.choice(rooms)
        goal = goal + np.random.uniform(size=2, low=-1, high=1)

        goal_state[:2] = goal
        opt_path = self.graph.find_path_to_closest_node(init_pos_idx, goal, self.value_policy, closest_node_mode="by_distance")
        endnode_of_optpath = self.graph.vertices[opt_path[0]]

        reached_endnode_of_optpath = False
        remaining_path = opt_path[:]
        done = False

        for t in range(self.max_path_length):

            high_level_actionidx = 0
            current_node = self.get_node_by_state(current_state)

            if endnode_of_optpath == current_node:
                reached_endnode_of_optpath = True

            dist_to_pathnodes = []
            for node_idx in remaining_path:
                dist_to_pathnodes.append((node_idx, distance(self.graph.vertices[node_idx], current_node)))
            dist_to_pathnodes.sort(key=lambda x: x[1])
            closest_idx, closest_dist = dist_to_pathnodes[0]

            if reached_endnode_of_optpath == False:
                if closest_dist != 0:
                    # node is off path
                    target_node_idx, target_node_dist = dist_to_pathnodes[0]
                    _, high_level_actionidx = self.assign_highlevel_action(current_node, self.graph.vertices[target_node_idx])
                else:
                    ###
                    # if current state is contained with in a node and the node has children which is closer to goal,
                    # take the action on the first edge of the path
                    ###
                    idx_path = remaining_path.index(closest_idx)
                    remaining_path = remaining_path[:idx_path]
                    _, high_level_actionidx = self.assign_highlevel_action(current_node, self.graph.vertices[remaining_path[-1]])
            else:
                _, high_level_actionidx = self.assign_highlevel_action(current_node, goal)

            states.append(current_state)
            previous_state = current_state

            actions.append(high_level_actionidx)

            # execute action on current state
            current_state, _, _, _ = self.env.step(high_level_actionidx)

            if done == False:
                done = True if np.linalg.norm((states[-1][:2] - goal), axis=-1) < self.goal_threshold else False

        trajectory_length = len(states)
        assert trajectory_length == self.max_path_length

        return np.stack(states), np.array(actions), goal_state, trajectory_length, opt_path, done

    def assign_highlevel_action(self, current_node, next_node):
        diff = np.subtract(np.array(next_node), np.array(current_node))
        # ['up', 'down', 'left', 'right']
        action = np.array([0.0, 0.0, 0.0, 0.0])
        if diff[0] >= 0 and diff[1] >= 0:  # go down_right
            action[3] = diff[0]
            action[1] = diff[1]
        elif diff[0] >= 0 and diff[1] < 0:  # go up_right
            action[3] = diff[0]
            action[0] = abs(diff[1])
        elif diff[0] < 0 and diff[1] >= 0:  # go down_left
            action[2] = abs(diff[0])
            action[1] = diff[1]
        else:  # go up_left
            action[2] = abs(diff[0])
            action[0] = abs(diff[1])

        action = action/float(max(abs(diff[0]), abs(diff[1])))
        high_level_actionidx = self.env.get_actionidx_by_action(action)
        return action, high_level_actionidx

    def take_policy_step(self, buffer=None):
        if buffer is None:
            buffer = self.replay_buffer

        avg_loss = 0
        self.policy_optimizer.zero_grad()

        for _ in range(self.n_accumulations):
            observations, actions, goals, _, horizons, weights = buffer.sample_batch(
                self.batch_size)
            loss = self.loss_fn(observations, goals,
                                actions, horizons, weights)
            loss.backward()
            avg_loss += ptu.to_numpy(loss)

        self.policy_optimizer.step()

        return avg_loss / self.n_accumulations

    def validation_loss(self, buffer=None):
        if buffer is None:
            buffer = self.validation_buffer

        if buffer is None or buffer.current_buffer_size == 0:
            return 0

        avg_loss = 0
        for _ in range(self.n_accumulations):
            observations, actions, goals, lengths, horizons, weights = buffer.sample_batch(
                self.batch_size)
            loss = self.loss_fn(observations, goals,
                                actions, horizons, weights)
            avg_loss += ptu.to_numpy(loss)

        return avg_loss / self.n_accumulations

    def loss_fn(self, observations, goals, actions, horizons, weights):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if self.is_discrete_action else torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype).cuda()
        goals_torch = torch.tensor(goals, dtype=obs_dtype).cuda()
        actions_torch = torch.tensor(actions, dtype=action_dtype).cuda()
        horizons_torch = torch.tensor(horizons, dtype=obs_dtype).cuda()
        weights_torch = torch.tensor(weights, dtype=torch.float32).cuda()

        conditional_nll = self.policy.nll(
            observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
        nll = conditional_nll

        return torch.mean(nll * weights_torch)
    
    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, opt_path=[], states=[], count=-1):
        '''
        1. object of rrt tree
        2. Plot RRT, obstacles and shortest path
        '''

        with open(osp.join(logger.get_snapshot_dir(), 'RRT_star_tree.pkl'), 'wb') as f:
            pickle.dump(self.graph, f)

        self.env.save_rrt_star_tree(timestep, goal_pos_x, goal_pos_y, goal_scatter_num,
                                    self.graph, opt_path=opt_path, states=states, count=count)

    def save_loss_fig(self, name, x, loss, validation_loss=None):
        plt.clf()
        fig, ax = plt.subplots()
        loss_line, = ax.plot(x, loss, color="b")
        loss_line.set_label('loss')
        if validation_loss != None:
            validation_loss_line, = ax.plot(x, validation_loss, color="r")
            validation_loss_line.set_label("validation_loss")
        ax.legend()
        ax.axis('on')
        plt.savefig("./code/gcsl_ant/fig/" + str(name) + "_loss.pdf")

    def get_node_by_state(self, state):
        return (((state[0] + 0.5 * self.rrt_tree_node_cover_size) // self.rrt_tree_node_cover_size)*self.rrt_tree_node_cover_size,
                ((state[1] + 0.5 * self.rrt_tree_node_cover_size) // self.rrt_tree_node_cover_size)*self.rrt_tree_node_cover_size)