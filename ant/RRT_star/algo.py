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
from gcsl import envs
from gcsl.envs.env_utils import DiscretizedActionEnv
import warnings

from RRT_star.utils import *
from RRT_star.v_function_core import *
try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_enabled = True
except:
    print('Tensorboard not installed!')
    tensorboard_enabled = False


class RRTStar_GCSL:
    def __init__(self,
                 env_name,
                 env,
                 env_for_checking,
                 tree,
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
        self.tree = tree

        self.policy = policy

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

        self.value_policy = value_policy

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

    def policy_accuracy(self):
        buffer = self.validation_buffer
        accuracys = []
        avg_loss = 0
        for _ in tqdm.tqdm(range(400)):
            observations, actions, goals, _, horizons, weights = buffer.sample_batch(
                self.batch_size)
            loss = self.loss_fn(observations, goals,
                                actions, horizons, weights)
            avg_loss += ptu.to_numpy(loss)

            obs_dtype = torch.float32
            action_dtype = torch.int64 if self.is_discrete_action else torch.float32

            observations_torch = torch.tensor(
                observations, dtype=obs_dtype).cuda()
            goals_torch = torch.tensor(goals, dtype=obs_dtype).cuda()
            actions_torch = torch.tensor(actions, dtype=action_dtype).cuda()
            horizons_torch = torch.tensor(horizons, dtype=obs_dtype).cuda()
            logits = self.policy.act_vectorized(
                observations_torch, goals_torch, horizon=horizons_torch, greedy=True, noise=0)

            # noisy_logits = logits * (1 - 0)
            # probs = torch.softmax(noisy_logits, 1)
            # samples = torch.argmax(probs, dim=-1)
            # samples = ptu.to_numpy(samples)
            # random_samples = np.random.choice(self.policy.action_space.n, size=len(samples))
            # predict = np.where(np.random.rand(len(samples)) < 0,random_samples,samples,)

        avg_loss = avg_loss/400
        accuracy = 0.0
        for i in range(len(actions_torch)):
            if logits[i] == actions_torch[i]:
                accuracy += 1

        accuracy = accuracy/len(actions_torch)
        accuracys.append(accuracy)

        # with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
        #     f.write(str(logits)+"\n")
        #     f.write(str(actions_torch)+"\n")
        #     f.write(str(len(logits))+"\n")
        #     f.write(str(len(actions_torch))+"\n")
        #     f.write(str(avg_loss)+"\n")
        # f.close()
        return sum(accuracys)/len(accuracys)

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
        # conditional_nll = self.policy.nll(
        #     observations_torch, goals_torch, actions_torch, horizon=None)
        nll = conditional_nll

        return torch.mean(nll * weights_torch)

    def search_covering_node(self, root_idx, state_node):
        # nodeidx = -1
        # node = []
        # for idx, v in enumerate(self.tree.vertices):
        #     if v == state_node_idx:
        #         node.append(v)
        #         nodeidx = idx
        #         break

        nodeidx = -1

        queue = deque()
        queue.append(root_idx)
        while len(queue) > 0:
            vertex_idx = queue.popleft()
            if self.tree.vertices[vertex_idx] == state_node:
                nodeidx = vertex_idx
                break
            for childidx, Child_cost in self.tree.children[vertex_idx]:
                queue.append(childidx)

        return nodeidx

    def minCostNearbyVex(self, newvex, radius, root_idx):
        # Nvex = ()
        # Nidx = -1
        # minDist = float("inf")

        queue = deque()
        queue.append(root_idx)
        res = []

        while len(queue) > 0:
            vertex_idx = queue.popleft()
            vertex_node = self.tree.vertices[vertex_idx]
            dist = distance(newvex, vertex_node)

            for childidx, Child_cost in self.tree.children[vertex_idx]:
                queue.append(childidx)

            if dist >= radius:
                continue

            line = Line(newvex, vertex_node)
            # if self.env.intersect_with_obstacle(line):
            #     continue

            dist += self.tree.distances[vertex_idx]
            res.append((vertex_idx, dist))
            # if dist < minDist:
            #     minDist = dist
            #     Nidx = vertex_idx
            #     Nvex = vertex_node

        res.sort(key=lambda x: x[1])
        return res

    # def save_waypoints(self, waypoints, prefix):
    #     self.env.save_waypoints(waypoints, prefix)
    #     return

    def sample_trajectory_for_evaluation(self, greedy=False, noise=0, render=False, goal_env="none"):
        if goal_env == "antumaze":
            goal_state = np.zeros(30)
            goal_state[:2] = np.array([0, 16])
            # goal_state[:2] = np.array([8, 17])
            goal = self.env.extract_goal(goal_state)
        elif goal_env == "antfall":
            goal_state = np.zeros(33)
            goal_state[:2] = np.array([0, 27])
            goal = self.env.extract_goal(goal_state)
        elif goal_env == "antpush":
            goal_state = np.zeros(30)
            goal_state[:2] = np.array([0, 19])
            goal = self.env.extract_goal(goal_state)
        elif goal_env == "antfourrooms":
            goal_state = np.zeros(30)
            goal_state[:2] = np.array([12, 12])
            goal = self.env.extract_goal(goal_state)
        elif goal_env == "antlongumaze":
            goal_state = np.zeros(30)
            goal_state[:2] = np.array([5, 0])
            goal = self.env.extract_goal(goal_state)
        elif goal_env == "antsmaze":
            goal_state = np.zeros(30)
            goal_state[:2] = np.array([12, 12])
            goal = self.env.extract_goal(goal_state)
        elif goal_env == "antpimaze":
            goal_state = np.zeros(30)
            goal_state[:2] = np.array([5, 0])
            goal = self.env.extract_goal(goal_state)
        else:
            goal_state = self.env.sample_goal()
            goal = self.env.extract_goal(goal_state)

        state = self.env.reset()

        if self.env_name in ['ant9rooms', 'ant9roomscloseddoors', 'ant16rooms', 'ant16roomscloseddoors']:
            # random_start_state = self.env.sample_goal()
            # random_start_pos = self.env.extract_goal(random_start_state)
            # state = self.env.reset_with_init_pos_at(
            #     random_start_pos)

            # # try only picking goal at the center of rooms
            # rooms = [[0, 0], [0, 18], [0, 27],
            #          [9, 0], [9, 9], [9, 18], [9, 27],
            #          [18, 0], [18, 9], [18, 18], [18, 27],
            #          [27, 0], [27, 9],  [27, 27]]
            # goal = np.array(random.choice(rooms))
            # goal_state[:2] = goal

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

            # try only picking goal at the center of rooms
            goal_state = self.env.sample_goal()

            goal = random.choice(rooms)
            while goal == start_pos:
                goal = random.choice(rooms)
            goal = goal + np.random.uniform(size=2, low=-1, high=1)

            goal_state[:2] = goal

            # [mws] special case testing
            state = self.env.reset_with_init_pos_at([18, 0])
            goal = np.array([0, 27])
            goal_state[:2] = goal

        # keep tracking of trajectory
        states = []
        actions = []

        for t in range(self.max_path_length):
            # if render:
            #     self.env.render()

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

            if done == True:
                # states.append(state)
                break
        trajectory_length = len(states)

        # pad the states and actions to be the same size (max_path_length * state.shape/action.shape)
        if trajectory_length < self.max_path_length:
            pad_len = self.max_path_length-trajectory_length
            pad_state = states[-1]
            pad_action = 0
            for i in range(pad_len):
                states.append(pad_state)
                actions.append(pad_action)

        return np.stack(states), np.array(actions), goal_state, trajectory_length

    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval', total_timesteps=0, env_type="None"):
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
            states, actions, goal_state, trajectory_len = self.sample_trajectory_for_evaluation(
                noise=0, greedy=greedy, goal_env=env_type)
            assert len(states) == len(actions)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(
                states[trajectory_len-1], goal_state)

            final_dist_vec[index] = final_dist

            success_vec[index] = (final_dist < self.goal_threshold)
            if success_vec[index] == True:
                avg_length += trajectory_len
                succeeded_count += 1
                if trajectory_len > max_length:
                    max_length = trajectory_len
        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            f.write(prefix + ' success ratio' + str(np.mean(success_vec))+"\n")
            f.write(prefix + ' avg final dist' +
                    str(np.mean(final_dist_vec))+"\n")
        f.close()
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

        return all_states, all_goal_states

    def visualize_evaluate_policy(self, eval_episodes, greedy=True, prefix='Eval', total_timesteps=0, env_type="none"):
        env = self.env

        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            f.write(env.maze_env._maze_id+'\n')
        f.close()
        avg_length = 0
        max_length = 0
        succeeded_count = 0
        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state, trajectory_len = self.sample_trajectory_for_evaluation(
                noise=0, greedy=greedy, goal_env=env_type)
            assert len(states) == len(actions)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(
                states[trajectory_len-1], goal_state)

            final_dist_vec[index] = final_dist

            with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
                f.write(str(trajectory_len)+"\n")
            f.close()
            success_vec[index] = (final_dist < self.goal_threshold)
            if success_vec[index] == True:
                avg_length += trajectory_len
                succeeded_count += 1
                if trajectory_len > max_length:
                    max_length = trajectory_len
        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            if succeeded_count > 0:
                f.write("Avg trace length till succeed:" +
                        str(avg_length / succeeded_count) + "\n")
                f.write("Max trace length till succeed:" +
                        str(max_length) + "\n")
        f.close()

        with open("./code/gcsl_ant/TraceData"+env_type+".txt", "a") as f:
            for (trace, goal, succeed) in zip(all_states, all_goal_states, success_vec):
                f.write(str([[list(state[:2])
                        for state in trace], list(goal[:2]), succeed])+"\n")
        f.close()

        print('%s/success ratio' % prefix,  np.mean(success_vec))
        print('%s avg final dist' % prefix,  np.mean(final_dist_vec))
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
            all_states, all_goal_states, success_vec)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s' % (prefix, key), value)

        return all_states, all_goal_states

    def visualize_evaluate_policy_with_LTL(self, abstract_LTL_graph, abstract_final_vertices, eval_episodes, greedy=True, prefix='Eval', total_timesteps=0, env_type="none", spec_num="", searchpolicy=False):
        env = self.env

        all_states = []
        all_goal_states = []
        all_actions = []
        all_reschedule_points = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            f.write(env.maze_env._maze_id+'\n')
        f.close()
        avg_length = 0
        max_length = 0
        succeeded_count = 0
        for index in tqdm.trange(eval_episodes, leave=True):

            # states, actions, goal_state, trajectory_len = self.sample_trajectory_for_evaluation_with_LTL(
            #     abstract_LTL_graph, abstract_final_vertices, noise=0, greedy=greedy, goal_env=env_type)
            states, actions, goal_state, trajectory_len, reschedule_points = self.sample_trajectory_for_evaluation_with_LTL_singlepath(
                abstract_LTL_graph, abstract_final_vertices, noise=0, greedy=greedy, goal_env=env_type)

            assert len(states) == len(actions)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            all_reschedule_points.append(reschedule_points)

            final_dist = env.goal_distance(
                states[trajectory_len-1], goal_state)

            final_dist_vec[index] = final_dist

            # with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            #     f.write(str(trajectory_len)+"\n")
            #     dist = 0
            #     for i in range(trajectory_len):
            #         if i >= 1:
            #             dist += distance(states[i][:2], states[i-1][:2])
            #     dist = dist/trajectory_len
            #     f.write('avgdist:'+str(dist)+'\n')
            # f.close()
            success_vec[index] = (final_dist < self.goal_threshold)
            if success_vec[index] == True:
                avg_length += trajectory_len
                succeeded_count += 1
                if trajectory_len > max_length:
                    max_length = trajectory_len
            # with open("./code/gcsl_ant/TraceData"+env_type+".txt", "a") as f:
            #     f.write(str([[list(state[:2])
            #             for state in states], list(goal_state[:2]), success_vec[-1],trajectory_len])+"\n")
            # f.close()

        with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            if succeeded_count != 0:
                f.write("Avg trace length till succeed:" +
                        str(avg_length / succeeded_count) + "\n")
            else:
                f.write("Avg trace length till succeed:" + str(0) + "\n")
            f.write("Max trace length till succeed:" + str(max_length) + "\n")
            f.write('/success ratio'+str(np.mean(success_vec)))
            f.write('avg final dist' + str(np.mean(final_dist_vec)))
        f.close()

        print('%s/success ratio' % prefix,  np.mean(success_vec))
        print('%s avg final dist' % prefix,  np.mean(final_dist_vec))
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

        # comment out to save time
        # diagnostics = env.get_diagnostics(
        #     all_states, all_goal_states, success_vec, spec_num, all_reschedule_points=all_reschedule_points)
        # for key, value in diagnostics.items():
        #     logger.record_tabular('%s %s' % (prefix, key), value)

        return all_states, all_goal_states

    def sample_trajectory_for_evaluation_with_LTL(self, abstract_LTL_graph, abstract_final_vertices, greedy=True, noise=0, render=False, goal_env="none"):

        # get sequence of tasks that having smallest accumulated '1-v' vlaue (dijkstra algo)
        num_nodes = len(abstract_LTL_graph)

        target_map = {0: (0, 0)}
        # [mws] special case
        # target_map = {0: (1, 2)}

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

                # if value > 0:
                #     value = 0
                # weight = cost - value
                weight = 1 - value
                if weight < 0:
                    weight = 0
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

        # # ablation study: random pick edges:
        # cur_vertex = 0
        # subtasks_queue = deque()

        # while cur_vertex not in abstract_final_vertices:
        #     cur_vertex = abstract_LTL_graph[cur_vertex][np.random.randint(len(abstract_LTL_graph[cur_vertex]))].target
        #     subtasks_queue.append(cur_vertex)
        # max_trajectory_len = 23*self.max_path_length

        subgoals = deque()
        for subtask in subtasks_queue:
            subgoals.append(
                np.array(target_map[subtask]) * self.env.room_scaling)

        max_trajectory_len = len(subtasks_queue)*self.max_path_length
        # print("max_trajectory_len:"+str(max_trajectory_len))
        # keep tracking of trajectory
        states_of_alltasks = []
        actions_of_alltasks = []

        # for i in range(len(subgoals)):
        #     print(str(subgoals[i])+'\n')

        # states_of_alltasks, actions_of_alltasks, goal_state, trajectory_length, opt_path = self.subtask_heuristic_search_LTL(
        #     subgoals)
        # return states_of_alltasks, actions_of_alltasks, goal_state, trajectory_length

        goal_state = self.env.sample_goal()
        state = self.env.reset()

        # only for a weird test
        # state = self.env.reset_with_init_pos_at([18, 0])

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

    def save_rrt_star_tree(self, timestep, goal_pos_x, goal_pos_y, goal_scatter_num, opt_path=[], states=[], count=-1):
        '''
        1. object of rrt tree
        2. Plot RRT, obstacles and shortest path
        '''

        with open(osp.join(logger.get_snapshot_dir(), 'RRT_star_tree.pkl'), 'wb') as f:
            pickle.dump(self.tree, f)

        self.env.save_rrt_star_tree(timestep, goal_pos_x, goal_pos_y, goal_scatter_num,
                                    self.tree, opt_path=opt_path, states=states, count=count)

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

    def grow_RRT_star_tree_using_graph(self):
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
                states, _, goal_state, _, opt_path, loss_v = self.sample_trajectory_for_growing_tree_using_Graph(
                    total_timesteps)

                ###
                sample_goals_x.append(goal_state[0])
                sample_goals_y.append(goal_state[1])
                ###
                if running_loss == None:
                    running_loss = loss_v
                else:
                    running_loss = 0.9 * running_loss + 0.1 * loss_v
                fig_loss_x.append(total_timesteps)
                fig_loss_y.append(running_loss)

                # self.save_rrt_star_tree(total_timesteps, sample_goals_x, sample_goals_y, 1, opt_path, states)

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

            with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
                f.write("rollouts total time (s) %d\n" %
                        (time.time() - start_time))
            f.close()
        self.save_rrt_star_tree(
            total_timesteps, sample_goals_x, sample_goals_y, 1, opt_path, states)
        self.save_loss_fig("value_policy_loss", fig_loss_x, fig_loss_y)
        self.env.sample_goal_scatter_fig(sample_goals_x, sample_goals_y)

    def save_value_policy_map(self, goal, timesteps):
        plt.clf()
        fig, ax = plt.subplots()
        ax = Draw_GridWorld(ax, self.env.maze_structure,
                            self.env.maze_env.MAZE_SIZE_SCALING)
        px = []
        py = []
        pv = []
        pv_max = -float('inf')
        pv_min = float('inf')
        for x in range(35):
            for y in range(35):
                state_node = [x-3, y-3]
                px.append(x + 1)
                py.append(y + 1)
                v = self.value_policy.get_value(state_node, goal)
                if v > pv_max:
                    pv_max = v
                if v < pv_min:
                    pv_min = v
                pv.append(v)

        for i in range(len(pv)):
            tmp = str(hex(int((pv[i]-pv_min)/(pv_max-pv_min)*255)))[2:]
            if len(tmp) == 1:
                tmp = '0' + tmp
            pv[i] = '#'+tmp*3
            # with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
            #     f.write(tmp)

        ax.scatter(px, py, c=pv)

        ax.scatter(goal[0] + 4 * self.env.maze_env.MAZE_SIZE_SCALING,
                   goal[1] + 4 * self.env.maze_env.MAZE_SIZE_SCALING, s=100, marker='*')

        plt.savefig("./code/gcsl_ant/fig/" + self.env.maze_env._maze_id +
                    "value_map" + str(timesteps) + str(goal)+".pdf")
        return

    # def save_value_policy_map_v2(self, state_node, timesteps):
    #     plt.clf()
    #     fig, ax = plt.subplots()
    #     ax = Draw_GridWorld(ax, self.env.maze_structure,
    #                         self.env.maze_env.MAZE_SIZE_SCALING)

    #     px = []
    #     py = []
    #     pv = []
    #     pv_max = -float('inf')
    #     pv_min = float('inf')
    #     for x in range(35):
    #         for y in range(35):
    #             goal = [x-3, y-3]
    #             px.append(x + 1)
    #             py.append(y + 1)
    #             v = self.value_policy.get_value(state_node, goal)
    #             if v > pv_max:
    #                 pv_max = v
    #             if v < pv_min:
    #                 pv_min = v
    #             pv.append(v)

    #     for i in range(len(pv)):
    #         tmp = str(hex(int((pv[i]-pv_min)/(pv_max-pv_min)*255)))[2:]
    #         if len(tmp) == 1:
    #             tmp = '0' + tmp
    #         pv[i] = '#'+tmp*3

    #     ax.scatter(px, py, c=pv)

    #     ax.scatter(state_node[0] + 4 * self.env.maze_env.MAZE_SIZE_SCALING,
    #                state_node[1] + 4 * self.env.maze_env.MAZE_SIZE_SCALING, s=100, marker='*')

    #     plt.savefig("./code/gcsl_ant/fig/" + self.env.maze_env._maze_id +
    #                 "value_map_v2" + str(timesteps) + str(state_node)+".pdf")
    #     return

    def sample_trajectory_for_growing_tree_using_Graph(self, total_timesteps):
        """
        test for sample trace using graph
        """
        goal_state = self.env.sample_goal()
        goal = self.env.extract_goal(goal_state)

        # keep tracking of trajectory
        states = []
        actions = []
        opt_path = []

        # self.save_value_policy_map(goal, total_timesteps)
        opt_path = self.tree.find_path_to_closest_node(
            self.tree.add_vex((0, 0)), goal, self.value_policy, closest_node_mode='by_value')
        loss_v = None

        if self.env_name in ["antpush"]:
            # states, actions, opt_path, extended, dataset = self.execute_tree_path_for_growing_tree_using_graph(goal, opt_path, total_timesteps)
            # loss_v = self.value_policy.execute(dataset)
            count = 0
            extended = False
            while len(opt_path) >= 1:
                states, actions, opt_path, extended, dataset, penalty = self.execute_tree_path_for_growing_tree_using_graph(
                    goal, opt_path, total_timesteps, count=count)
                loss_v = self.value_policy.execute(
                    dataset, penalty, self.rrt_tree_node_cover_size)
                count += 1
                if extended == True:  # or count > 10:
                    break

                # self.save_rrt_star_tree(
                #     total_timesteps + count, [goal[0]], [goal[1]], 1, opt_path, states)
                # self.save_value_policy_map(goal, total_timesteps + count)
                opt_path = self.tree.find_path_to_closest_node(
                    self.tree.add_vex((0, 0)), goal, self.value_policy, closest_node_mode='by_value')
        else:
            states, actions, opt_path, extended, dataset, penalty = self.execute_tree_path_for_growing_tree_using_graph(
                goal, opt_path, total_timesteps)
            loss_v = self.value_policy.execute(
                dataset, penalty, self.rrt_tree_node_cover_size)

        trajectory_length = len(states)

        return np.stack(states), np.array(actions), goal_state, trajectory_length, opt_path, loss_v

    def execute_tree_path_for_growing_tree_using_graph(self, goal, opt_path, total_timesteps, count=-1):

        states = []
        actions = []
        init_state = self.env.reset()
        current_state = init_state
        previous_state = current_state
        current_node = self.get_node_by_state(current_state)
        prev_node = current_node

        # add init state node into tree(centered [0,0])
        init_node = self.get_node_by_state(current_state)
        init_pos_idx = self.tree.add_vex(init_node)

        remaining_path = opt_path[:]
        endnode_of_optpath = self.tree.vertices[opt_path[0]]
        reached_endnode_of_optpath = False

        done = False
        extended = False
        dataset = []

        for t in range(self.max_path_length):
            high_level_actionidx = 0
            current_node = self.get_node_by_state(current_state)
            prev_node = self.get_node_by_state(previous_state)
            try:
                prev_nodeidx = self.tree.vex2idx[prev_node]
            except:
                prev_nodeidx = -1

            if endnode_of_optpath == current_node:
                reached_endnode_of_optpath = True

            # search state node in tree that covers the current_state
            current_node_idx = self.search_covering_node_for_graph(
                init_pos_idx, current_node)

            dist_to_pathnodes = []
            for node_idx in remaining_path:
                dist_to_pathnodes.append(
                    (node_idx, distance(self.tree.vertices[node_idx], current_node)))
            dist_to_pathnodes.sort(key=lambda x: x[1])
            closest_idx, closest_dist = dist_to_pathnodes[0]

            if prev_nodeidx > -1:
                if current_node_idx > -1:
                    # if current_node is contained in some state, add an new edge into the tree if distance cost is smaller and new path is feasible
                    dist = distance(current_node, prev_node)
                    if prev_nodeidx != current_node_idx:
                        self.tree.add_edge(
                            current_node_idx, prev_nodeidx, dist)
                else:
                    ###
                    # if:
                    # 1. current state is not covered by any node in the tree
                    # 2. previous state is covered by some node,
                    # then add new node and update previous node and edge
                    ###
                    current_node_idx = self.tree.add_vex(current_node)
                    dist = distance(current_node, prev_node)
                    self.tree.add_edge(
                        current_node_idx, prev_nodeidx, dist)
                    extended = True

            if done == True:
                # states.append(state)
                extended = True
                break

            if reached_endnode_of_optpath == False:
                if closest_dist != 0:
                    high_level_action, high_level_actionidx = self.assign_highlevel_action(
                        current_node, self.tree.vertices[closest_idx])
                else:
                    idx_path = remaining_path.index(closest_idx)
                    remaining_path = remaining_path[:idx_path]
                    high_level_action, high_level_actionidx = self.assign_highlevel_action(
                        current_node, self.tree.vertices[remaining_path[-1]])
            else:
                high_level_action, high_level_actionidx = self.assign_highlevel_action(
                    current_node, goal)

            high_level_actionidx = int(high_level_actionidx)
            states.append(current_state)
            previous_state = current_state
            actions.append(high_level_actionidx)

            # execute action on current state
            current_state, _, _, _ = self.env.step(
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
            dataset[-1]['reward'] = 1

        penalty = False
        if done == False and extended == False:
            penalty = True

        return states, actions, opt_path, extended, dataset, penalty

    def fine_tuning_value_policy(self):
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

        test_points_9rooms = [[1, 1], [0, 8], [0, 18], [9, 0],
                              [9, 9], [9, 18], [18, 0], [18, 9], [18, 18]]

        test_points_16rooms = [[1, 1], [0, 8], [0, 18], [0, 27],
                               [9, 0], [9, 9], [9, 18], [9, 27],
                               [18, 0], [18, 9], [18, 18], [18, 27],
                               [27, 0], [27, 9], [27, 18], [27, 27]]
        case_study_envs_9rooms = ['ant9rooms', 'ant9roomscloseddoors']
        case_study_envs_16rooms = ['ant16rooms', 'ant16roomscloseddoors']

        # if self.env_name in case_study_envs_9rooms:
        #     for point in test_points_9rooms:
        #         self.graph_point_test(str(point), (0, 0), point)

        # if self.env_name in case_study_envs_16rooms:
        #     for point in test_points_16rooms:
        #         self.graph_point_test(str(point), (0, 0), point)

        total_trace = 0
        succeeded_trace = 0
        fig_loss_x = []
        fig_loss_y = []
        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            # while total_timesteps < self.max_timesteps:
            while total_timesteps < int(self.max_timesteps):
                # Interact in environmenta according to exploration strategy.
                states, actions, goal_state, trajectory_len, opt_path, dataset, penalty = self.sample_trajectory_for_fine_tuning_buffer()
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

                # self.save_rrt_star_tree(
                #     total_timesteps, sample_goals_x, sample_goals_y, 1, opt_path, states)

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
            with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
                f.write("[fine tune] rollouts total time (s) %d\n" %
                        (time.time() - start_time))
                f.write("[fine tune] # of traces:"+str(total_trace)+"\n")
                f.write("[fine tune] # of succeed traces:" +
                        str(succeeded_trace)+"\n")
                f.write("[fine tune] goal reaching success rate:" +
                        str(succeeded_trace / total_trace)+"\n")
            f.close()

        if self.env_name in case_study_envs_9rooms:
            for point in test_points_9rooms:
                self.graph_point_test(
                    str(total_timesteps)+str(point), (0, 0), point)

        if self.env_name in case_study_envs_16rooms:
            for point in test_points_16rooms:
                self.graph_point_test(
                    str(total_timesteps)+str(point), (0, 0), point)

    def test_fine_tuned_value_policy(self, start, goal):
        if logger.get_snapshot_dir() and self.log_tensorboard:
            self.summary_writer = SummaryWriter(
                osp.join(logger.get_snapshot_dir(), 'tensorboard'))

        test_points_9rooms = [[1, 1], [0, 8], [0, 18], [9, 0],
                              [9, 9], [9, 18], [18, 0], [18, 9], [18, 18]]

        test_points_16rooms = [[1, 1], [0, 8], [0, 18], [0, 27],
                               [9, 0], [9, 9], [9, 18], [9, 27],
                               [18, 0], [18, 9], [18, 18], [18, 27],
                               [27, 0], [27, 9], [27, 18], [27, 27]]
        case_study_envs_9rooms = ['ant9rooms', 'ant9roomscloseddoors']
        case_study_envs_16rooms = ['ant16rooms', 'ant16roomscloseddoors']

        # if self.env_name in case_study_envs_9rooms:
        #     for point in test_points_9rooms:
        #         self.graph_point_test(str(point), start, point)

        # if self.env_name in case_study_envs_16rooms:
        #     for point in test_points_16rooms:
        #         self.graph_point_test(str(point), start, point)
        value = self.value_policy.get_value(start, goal)
        print('From: '+str(start)+" To: " +
              str(goal)+" Value: " + str(value)+'\n')

    def graph_point_test(self, timestep, start, goal):
        '''
        must be changed when changing env
        '''
        # self.save_value_policy_map(goal, total_timesteps)
        opt_path = self.tree.find_path_to_closest_node(
            self.tree.vex2idx[tuple(start)], goal, self.value_policy, closest_node_mode="by_distance")
        # opt_path = self.tree.find_path_to_closest_node(
        #     self.tree.vex2idx[tuple(start)], goal, self.value_policy, closest_node_mode="by_value")
        self.save_rrt_star_tree(timestep, [goal[0]], [goal[1]], 1, opt_path)
        self.save_value_policy_map(goal, timestep)
        # self.save_value_policy_map_v2(goal, timestep)

    def search_covering_node_for_graph(self, root_idx, state_node):
        nodeidx = -1

        for i in range(len(self.tree.vertices)):
            if self.tree.vertices[i] == state_node:
                nodeidx = i
                break
        return nodeidx

    def train_SL_using_graph(self, samples_num, training_times):
        start_time = time.time()
        last_time = start_time
        sample_time = 0

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

        sampling_time = time.time()
        timesteps_train = samples_num/training_times
        total_trace = 0
        succeeded_trace = 0
        with tqdm.tqdm(total=samples_num, smoothing=0) as ranger:
            for n in range(samples_num):
                states, actions, goal_state, trajectory_len, opt_path, done = self.sample_trajectory_for_SLbuffer_using_graph()
                total_trace += 1
                if done:
                    succeeded_trace += 1
                    # self.save_rrt_star_tree(n, [goal_state[0]], [
                    #                         goal_state[1]], 1, opt_path, states)
                    assert len(states) == len(actions)
                    self.replay_buffer.add_trajectory(
                        states, actions, goal_state, length_of_traj=trajectory_len)
                    if self.validation_buffer is not None and np.random.rand() < 0.2:
                        self.validation_buffer.add_trajectory(
                            states, actions, goal_state, length_of_traj=trajectory_len)
                    # else:
                    #     self.replay_buffer.add_trajectory(
                    #         states, actions, goal_state, length_of_traj=trajectory_len)
                ranger.update(1)

            with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
                f.write("[SL] SL trace collecting time(s) %d\n" %
                        (time.time() - sampling_time))
            f.close()

        # # save validation buffer
        # self.validation_buffer.save(
        #     osp.join(logger.get_snapshot_dir()+'/validation_buffer.pkl'))
        # # save training buffer
        # self.replay_buffer.save(
        #     osp.join(logger.get_snapshot_dir()+'/replay_buffer.pkl'))

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
                        # running_loss = 0.9 * running_loss + 0.1 * loss
                        running_loss += loss

                    if running_validation_loss is None:
                        running_validation_loss = validation_loss
                    else:
                        # running_validation_loss = 0.9 * running_validation_loss + 0.1 * validation_loss
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

                # # Logging Code
                # if logger.get_snapshot_dir():
                #     modifier = str(x) if self.save_every_iteration else ''
                #     torch.save(
                #         self.policy.state_dict(),
                #         osp.join(logger.get_snapshot_dir(),
                #                  'policy%s.pkl' % modifier)
                #     )
                #     if hasattr(self.replay_buffer, 'state_dict'):
                #         with open(osp.join(logger.get_snapshot_dir(), 'buffer%s.pkl' % modifier), 'wb') as f:
                #             pickle.dump(self.replay_buffer.state_dict(), f)
                #     if hasattr(self.replay_buffer, 'state_dict'):
                #         with open(osp.join(logger.get_snapshot_dir(), 'validation_buffer%s.pkl' % modifier), 'wb') as f:
                #             pickle.dump(self.validation_buffer.state_dict(), f)

                #     full_dict = dict(env=self.env, policy=self.policy)
                #     with open(osp.join(logger.get_snapshot_dir(), 'params%s.pkl' % modifier), 'wb') as f:
                #         pickle.dump(full_dict, f)

                # save model after training 50*1000 batches and after training 200*1000 batches
                if (x + 1) % 50 == 0:
                    self.evaluate_policy(
                        eval_episodes=20, greedy=True, prefix='Eval', env_type=self.env_name)
                    accuracy = self.policy_accuracy()
                    with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
                        f.write(
                            "[TEST] After %s iterations of training:accuracy %s \n" % ((x+1), accuracy))
                    f.close()
                    if logger.get_snapshot_dir():
                        modifier = str(x)
                        torch.save(
                            self.policy.state_dict(),
                            osp.join(logger.get_snapshot_dir(),
                                     'policy%s.pkl' % modifier)
                        )
                        # if hasattr(self.replay_buffer, 'state_dict'):
                        #     with open(osp.join(logger.get_snapshot_dir(), 'buffer%s.pkl' % modifier), 'wb') as f:
                        #         pickle.dump(self.replay_buffer.state_dict(), f)

                        full_dict = dict(env=self.env, policy=self.policy)
                        with open(osp.join(logger.get_snapshot_dir(), 'params%s.pkl' % modifier), 'wb') as f:
                            pickle.dump(full_dict, f)
                    self.save_loss_fig("SL_graph" + modifier, fig_loss_x,
                                       fig_loss_y, validation_loss=fig_validation_loss_y, )
                    with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
                        f.write("[SL] After %s iterations of training:\n" %
                                modifier)
                        f.write("[SL] SL graph training time (s) %d\n" % (
                            time.time() - start_out_of_rollout_training_time))
                        f.write("[SL] total time (s) %d\n" %
                                (time.time() - start_time))
                    f.close()

            with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
                f.write("[SL] # of traces:"+str(total_trace))
                f.write("[SL] # of succeed traces:"+str(succeeded_trace))
                f.write("[SL] goal reaching success rate:" +
                        str(succeeded_trace / total_trace)+"\n")
            f.close()

        self.save_loss_fig("SL_graph", fig_loss_x, fig_loss_y,
                           validation_loss=fig_validation_loss_y)
        # self.save_rrt_star_tree(0, [0], [0], 0, [])

    # , greedy=True, noise=0):
    def sample_trajectory_for_SLbuffer_using_graph(self):

        states = []
        actions = []

        # # set random start position
        # if self.env_name in ['ant9rooms', 'ant9roomscloseddoors', 'ant16rooms', 'ant16roomscloseddoors']:
        #     random_start_pos = self.tree.get_random_start_pos()
        #     current_state = self.env.reset_with_init_pos_at(
        #         random_start_pos)

        # previous_state = current_state

        # init_pos = self.get_node_by_state(current_state)
        # init_pos_idx = self.tree.vex2idx[init_pos]

        # endnode_of_optpath = init_pos

        # # choose only one from the two pieces of code below
        # # try only picking goal at the center of rooms
        # while endnode_of_optpath == init_pos:
        #     goal_state = self.env.sample_goal()
        #     goal = self.env.extract_goal(goal_state)
        #     opt_path = self.tree.find_path_to_closest_node(
        #         init_pos_idx, goal, self.value_policy, closest_node_mode="by_distance")
        #     endnode_of_optpath = self.tree.vertices[opt_path[0]]

        rooms = [[0, 0], [0, 18], [0, 27],
                 [9, 0], [9, 9], [9, 18], [9, 27],
                 [18, 0], [18, 9], [18, 18], [18, 27],
                 [27, 0], [27, 9],  [27, 27]]

        # for ant16rooms
        start_pos = random.choice(rooms)
        random_start_pos = start_pos + \
            np.random.uniform(size=2, low=-1, high=1)
        current_state = self.env.reset_with_init_pos_at(
            random_start_pos)

        previous_state = current_state

        init_pos = self.get_node_by_state(current_state)
        init_pos_idx = self.tree.vex2idx[init_pos]

        endnode_of_optpath = init_pos

        # try only picking goal at the center of rooms
        goal_state = self.env.sample_goal()

        goal = random.choice(rooms)
        while goal == start_pos:
            goal = random.choice(rooms)
        goal = goal + np.random.uniform(size=2, low=-1, high=1)

        goal_state[:2] = goal
        opt_path = self.tree.find_path_to_closest_node(
            init_pos_idx, goal, self.value_policy, closest_node_mode="by_distance")
        endnode_of_optpath = self.tree.vertices[opt_path[0]]

        reached_endnode_of_optpath = False
        remaining_path = opt_path[:]
        done = False

        for t in range(self.max_path_length):

            # if done == True:
            #     break

            high_level_actionidx = 0
            current_node = self.get_node_by_state(current_state)

            if endnode_of_optpath == current_node:
                reached_endnode_of_optpath = True

            dist_to_pathnodes = []
            for node_idx in remaining_path:
                dist_to_pathnodes.append(
                    (node_idx, distance(self.tree.vertices[node_idx], current_node)))
            dist_to_pathnodes.sort(key=lambda x: x[1])
            closest_idx, closest_dist = dist_to_pathnodes[0]

            if reached_endnode_of_optpath == False:
                if closest_dist != 0:
                    # node is off path
                    target_node_idx, target_node_dist = dist_to_pathnodes[0]
                    high_level_action, high_level_actionidx = self.assign_highlevel_action(
                        current_node, self.tree.vertices[target_node_idx])
                else:
                    ###
                    # if current state is contained with in a node and the node has children which is closer to goal,
                    # take the action on the first edge of the path
                    ###
                    idx_path = remaining_path.index(closest_idx)
                    remaining_path = remaining_path[:idx_path]
                    high_level_action, high_level_actionidx = self.assign_highlevel_action(
                        current_node, self.tree.vertices[remaining_path[-1]])
            else:
                high_level_action, high_level_actionidx = self.assign_highlevel_action(
                    current_node, goal)

            states.append(current_state)
            previous_state = current_state

            actions.append(high_level_actionidx)

            # execute action on current state
            current_state, _, _, _ = self.env.step(
                high_level_actionidx)

            if done == False:
                done = True if np.linalg.norm(
                    (states[-1][:2] - goal), axis=-1) < self.goal_threshold else False

        trajectory_length = len(states)
        assert trajectory_length == self.max_path_length

        return np.stack(states), np.array(actions), goal_state, trajectory_length, opt_path, done

    def sample_trajectory_for_fine_tuning_buffer(self):

        states = []
        actions = []

        # # random position
        # if self.env_name in ['ant9rooms', 'ant9roomscloseddoors', 'ant16rooms', 'ant16roomscloseddoors']:
        #     random_start_pos = self.tree.get_random_start_pos()
        #     current_state = self.env.reset_with_init_pos_at(
        #         random_start_pos)

        # previous_state = current_state

        # init_pos = self.get_node_by_state(current_state)
        # init_pos_idx = self.tree.vex2idx[init_pos]

        # endnode_of_optpath = init_pos

        # # try random goals over the space
        # while endnode_of_optpath == init_pos:
        #     goal_state = self.env.sample_goal()
        #     goal = self.env.extract_goal(goal_state)
        #     opt_path = self.tree.find_path_to_closest_node(
        #         init_pos_idx, goal, self.value_policy, closest_node_mode="by_distance")
        #     endnode_of_optpath = self.tree.vertices[opt_path[0]]

        rooms = [[0, 0], [0, 18], [0, 27],
                 [9, 0], [9, 9], [9, 18], [9, 27],
                 [18, 0], [18, 9], [18, 18], [18, 27],
                 [27, 0], [27, 9],  [27, 27]]

        # for ant16rooms
        start_pos = random.choice(rooms)
        random_start_pos = start_pos + \
            np.random.uniform(size=2, low=-1, high=1)
        current_state = self.env.reset_with_init_pos_at(
            random_start_pos)

        previous_state = current_state

        init_pos = self.get_node_by_state(current_state)
        init_pos_idx = self.tree.vex2idx[init_pos]

        endnode_of_optpath = init_pos

        # try only picking goal at the center of rooms
        goal_state = self.env.sample_goal()

        goal = random.choice(rooms)
        while goal == start_pos:
            goal = random.choice(rooms)
        goal = goal + np.random.uniform(size=2, low=-1, high=1)

        goal_state[:2] = goal
        opt_path = self.tree.find_path_to_closest_node(
            init_pos_idx, goal, self.value_policy, closest_node_mode="by_distance")
        endnode_of_optpath = self.tree.vertices[opt_path[0]]

        reached_endnode_of_optpath = False
        remaining_path = opt_path[:]
        done = False
        dataset = []

        for t in range(self.max_path_length):

            if done == True:
                break

            high_level_actionidx = 0
            current_node = self.get_node_by_state(current_state)

            if endnode_of_optpath == current_node:
                reached_endnode_of_optpath = True

            dist_to_pathnodes = []
            for node_idx in remaining_path:
                dist_to_pathnodes.append(
                    (node_idx, distance(self.tree.vertices[node_idx], current_node)))
            dist_to_pathnodes.sort(key=lambda x: x[1])
            closest_idx, closest_dist = dist_to_pathnodes[0]

            if reached_endnode_of_optpath == False:
                if closest_dist != 0:
                    # node is off path
                    target_node_idx, target_node_dist = dist_to_pathnodes[0]
                    high_level_action, high_level_actionidx = self.assign_highlevel_action(
                        current_node, self.tree.vertices[target_node_idx])
                else:
                    ###
                    # if current state is contained with in a node and the node has children which is closer to goal,
                    # take the action on the first edge of the path
                    ###
                    idx_path = remaining_path.index(closest_idx)
                    remaining_path = remaining_path[:idx_path]
                    high_level_action, high_level_actionidx = self.assign_highlevel_action(
                        current_node, self.tree.vertices[remaining_path[-1]])
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
        if done == False:
            # dataset[-1]['rew'] = -1
            penalty = True

        trajectory_length = len(states)

        return np.stack(states), np.array(actions), goal_state, trajectory_length, opt_path, dataset, penalty

    def subtask_heuristic_search_LTL(self, subgoals):

        states_of_alltasks = []
        actions_of_alltasks = []
        opt_path_of_alltasks = []
        max_trajectory_len = len(subgoals)*self.max_path_length

        goal_state = self.env.sample_goal()
        self.env.reset()
        prefix = len(subgoals)
        while len(subgoals) != 0:
            sub_goal = np.array(subgoals.popleft(
            ))
            goal_state[:2] = sub_goal

            current_state = self.env.maze_env._get_obs()
            previous_state = current_state

            init_pos = self.get_node_by_state(current_state)
            init_pos_idx = self.tree.vex2idx[init_pos]

            endnode_of_optpath = init_pos

            while endnode_of_optpath == init_pos:
                goal = self.env.extract_goal(goal_state)
                opt_path = self.tree.find_path_to_closest_node(
                    init_pos_idx, goal, self.value_policy, closest_node_mode="by_distance")
                endnode_of_optpath = self.tree.vertices[opt_path[0]]
            opt_path_of_alltasks += reversed(opt_path)

            reached_endnode_of_optpath = False
            remaining_path = opt_path[:]

            done = False
            states = []
            actions = []

            for t in range(self.max_path_length):
                if done == True:
                    break

                high_level_actionidx = 0
                current_node = self.get_node_by_state(current_state)

                if endnode_of_optpath == current_node:
                    reached_endnode_of_optpath = True

                dist_to_pathnodes = []
                for node_idx in remaining_path:
                    dist_to_pathnodes.append(
                        (node_idx, distance(self.tree.vertices[node_idx], current_node)))
                dist_to_pathnodes.sort(key=lambda x: x[1])
                closest_idx, closest_dist = dist_to_pathnodes[0]

                if reached_endnode_of_optpath == False:
                    if closest_dist != 0:
                        # node is off path
                        target_node_idx, target_node_dist = dist_to_pathnodes[0]
                        high_level_action, high_level_actionidx = self.assign_highlevel_action(
                            current_node, self.tree.vertices[target_node_idx])
                    else:
                        ###
                        # if current state is contained with in a node and the node has children which is closer to goal,
                        # take the action on the first edge of the path
                        ###
                        idx_path = remaining_path.index(closest_idx)
                        remaining_path = remaining_path[:idx_path]
                        high_level_action, high_level_actionidx = self.assign_highlevel_action(
                            current_node, self.tree.vertices[remaining_path[-1]])
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

            states_of_alltasks += states
            actions_of_alltasks += actions
            if done == False:
                break

        trajectory_length = len(states_of_alltasks)

        if trajectory_length < max_trajectory_len:
            pad_len = max_trajectory_len - trajectory_length
            pad_state = states_of_alltasks[-1]
            pad_action = 0
            for i in range(pad_len):
                states_of_alltasks.append(pad_state)
                actions_of_alltasks.append(pad_action)

        self.save_rrt_star_tree(prefix, [goal_state[0]], [
                                goal_state[1]], 1, opt_path_of_alltasks)

        return np.stack(states_of_alltasks), np.array(actions_of_alltasks), goal_state, trajectory_length, opt_path,

    def sample_trajectory_for_evaluation_with_LTL_singlepath(self, abstract_LTL_graph, abstract_final_vertices, greedy=True, noise=0, render=False, goal_env="none"):
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
                    room = edge.predicate(
                        (0, 0), [])[0][1], edge.predicate((0, 0), [])[0][0]
                    target_map[edge.target] = [
                        x * self.env.room_scaling for x in room]

        nodes = [i for i in range(num_nodes)]
        distance = []
        predecessor = []
        # initialize graph
        for i in range(num_nodes):
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

                cur_state = target_map[curNode]
                to_state = target_map[to_node]
                value = self.value_policy.get_value(
                    cur_state, to_state)

                weight = 1 - value
                if weight < 0:
                    weight = 0
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

        # subgoals = deque()
        # for subtask in subtasks_queue:
        #     subgoals.append(
        #         np.array(target_map[subtask]) * self.env.room_scaling)

        max_trajectory_len = len(subtasks_queue)*self.max_path_length
        # max_trajectory_len = 23 * self.max_path_length
        # print("max_trajectory_len:"+str(max_trajectory_len))

        # keep tracking of trajectory
        states_of_alltasks = []
        actions_of_alltasks = []

        goal_state = self.env.sample_goal()
        state = self.env.reset()
        reached_nodeidx = 0
        step_count = 0

        reschedule_points = [(0, 0)]

        while len(subtasks_queue) != 0:
            target_nodeidx = subtasks_queue.popleft()
            sub_goal = np.array(target_map[target_nodeidx])
            goal_state[:2] = sub_goal

            done = False
            states = []
            actions = []
            for t in range(self.max_path_length):
                if done == True:
                    break
                step_count += 1

                if step_count % 50 == 0:
                    # select a new path from abstract sub-graph
                    # 1) get sub-graph and sub-target map
                    reschedule_points.append(
                        tuple(self.get_node_by_state(state)))
                    nodes = []
                    target_map = {reached_nodeidx: tuple(
                        self.get_node_by_state(state))}
                    tmp = deque()
                    tmp.append(reached_nodeidx)
                    while len(tmp) != 0:
                        n = tmp.popleft()
                        nodes.append(n)
                        n_edges = abstract_LTL_graph[n]
                        for edge in n_edges:
                            if n != edge.target and target_map.get(edge.target) == None:
                                # adjust the form of room to fit ant env
                                room = edge.predicate(
                                    (0, 0), [])[0][1], edge.predicate((0, 0), [])[0][0]
                                target_map[edge.target] = [
                                    x * self.env.room_scaling for x in room]
                            tmp.count
                        for neig in neighbors[n]:
                            if n != neig and tmp.count(neig) < 1:
                                tmp.append(neig)

                    # 2) get next sub-goal
                    # initialize graph
                    distance = []
                    predecessor = []
                    for i in range(num_nodes):
                        distance.append(float('inf'))
                        predecessor.append(-1)

                    # dist=0 for the start node
                    distance[reached_nodeidx] = 0
                    # print(nodes)
                    # print(target_map)
                    while len(nodes) > 0:
                        curNode = min(nodes, key=lambda node: distance[node])
                        nodes.remove(curNode)
                        if distance[curNode] == float('inf'):
                            break
                        nodes.index
                        for to_node in neighbors[curNode]:

                            cur_state = target_map[curNode]
                            to_state = target_map[to_node]
                            value = self.value_policy.get_value(
                                cur_state, to_state)
                            weight = 1 - value
                            if weight < 0:
                                weight = 0
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

                    # print(subtasks_queue)
                    target_nodeidx = subtasks_queue.popleft()
                    sub_goal = np.array(target_map[target_nodeidx])
                    goal_state[:2] = sub_goal

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

                if np.linalg.norm((state[:2] - sub_goal), axis=-1) < self.goal_threshold:
                    done = True
                    reached_nodeidx = target_nodeidx

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
        return np.stack(states_of_alltasks), np.array(actions_of_alltasks), goal_state, trajectory_length, reschedule_points
