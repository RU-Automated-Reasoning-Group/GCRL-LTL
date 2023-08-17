import gym
import numpy as np

from spectrl.main.monitor import Resource_Model
from spectrl.rl.ars import NNPolicy, NNParams, ars
from spectrl.rl.ddpg import DDPG
from spectrl.util.dist import FiniteDistribution
from spectrl.util.rl import get_rollout, RandomPolicy
from heapq import heappop, heappush


class AbstractEdge:
    '''
    Class defining an abstract edge.
    Vertices are integers from 0 to |U|.

    Parameters:
        target: int (target vertex)
        predicate: state, resource -> float (predicate corresponding to target)
        constraints: list of constraints that needs to be maintained (one after the other)
    '''

    def __init__(self, target, predicate, constraints):
        self.target = target
        self.predicate = predicate
        self.constraints = constraints

    def learn_policy(self, env, hyperparams, source_vertex, init_dist=None,
                     algo='ars', res_model=None, max_steps=100, safety_penalty=-1,
                     neg_inf=-10, alpha=0, use_gpu=False, render=False):
        '''
        Learn policy for the abstract edge.

        Parameters:
            env: gym.Env (with additional method, set_state: np.array -> NoneType)
            init_dist: Distribution (initial state distribution)
            hyperparams: HyperParams object (corresponding to the learning algorithm)
            algo: str (RL algorithm to use)
            res_model: Resource_Model (optional)
            max_steps: int (maximum steps for an episode while training)

        Returns:
            Policy object with get_action(combined_state) function.
        '''

        # Step 1: Create reachability environment.
        reach_env = ReachabilityEnv(env, init_dist, self.predicate, self.constraints,
                                    max_steps=max_steps, res_model=res_model,
                                    safety_penalty=safety_penalty, neg_inf=neg_inf,
                                    alpha=alpha)

        # Step 2: Call the learning algorithm
        print('\nLearning policy for edge {} -> {}\n'.format(source_vertex, self.target))
        if self.constraints[0].__name__ == 'true_pred' and self.predicate is None:
            policy = RandomPolicy(reach_env.action_space.shape[0], reach_env.action_space.high)
            log_info = np.array([[0, 0, 0]])
        elif algo == 'ars':
            nn_params = NNParams(reach_env, hyperparams.hidden_dim)
            policy = NNPolicy(nn_params)
            log_info = ars(reach_env, policy, hyperparams.ars_params)
        elif algo == 'ddpg':
            agent = DDPG(hyperparams, use_gpu=use_gpu)
            agent.train(reach_env)
            policy = agent.get_policy()
            log_info = agent.rewardgraph
        else:
            raise ValueError('Algorithm \"{}\" not supported!'.format(algo))

        # Render for debugging
        if render:
            for _ in range(20):
                get_rollout(reach_env, policy, True)
            reach_env.close_viewer()

        return policy, reach_env, log_info


class AbstractReachability:
    '''
    Class defining the abstract reachability problem.

    Parameters:
        abstract_graph: list of list of abstract edges (adjacency list).
        final_vertices: set of int (set of final vertices).

    Initial vertex is assumed to be 0.
    '''

    def __init__(self, abstract_graph, final_vertices):
        self.abstract_graph = abstract_graph
        self.final_vertices = final_vertices
        self.num_vertices = len(self.abstract_graph)

    def learn_dijkstra_policy(self, env, hyperparams, algo='ars', res_model=None,
                              max_steps=100, safety_penalty=-1, neg_inf=-10, alpha=0,
                              num_samples=300, use_gpu=False, render=False, succ_thresh=0.):
        '''
        Dijkstra's algorithm based learning for abstract reachability.

        Parameters:
            env: gym.Env (with additional method, set_state: np.array -> NoneType)
            init_dist: Distribution (initial state distribution)
            hyperparams: HyperParams object (corresponding to the RL algo)
            algo: str (RL algorithm to use to learn policies for edges)
            res_model: Resource_Model (optional)
            safety_penalty: float (min penalty for violating constraints)
            neg_inf: float (large negative constant)
            num_samples: int (number of samples used to compute reach probabilities)

        Returns:
            abstract_policy: list of int (edge to take for each vertex)
            nn_policies: list of list of nn policies (can be [] for unexplored vertices or
                            None for edges for which no policy was learned)
        '''
        # Initialize abstract policy and NN policies.
        parent = [-1] * self.num_vertices
        abstract_policy = [-1] * self.num_vertices
        nn_policies = [[] for edges in self.abstract_graph]

        # Dijkstra initialization
        explored = [False] * self.num_vertices
        min_neg_log_prob = [np.inf] * self.num_vertices
        queue = []
        heappush(queue, (0, 0, -1))  # (distance, vertex, source) triples
        reached_final_vertex = False

        # Reach states for each vertex and source
        reach_states = {}
        success_measure = 0  # negative log of success probability
        num_edges_learned = 0
        total_steps = 0
        total_time = 0.

        # Set of bad edges for which RL fails to learn a policy
        bad_edges = []
        incomplete = True
        best_success = 0.

        while not reached_final_vertex:
            if len(queue) == 0:
                break

            neg_log_prob, vertex, source = heappop(queue)

            if not explored[vertex]:

                # Set minimum log probability of reaching the vertex and the last edge taken
                min_neg_log_prob[vertex] = neg_log_prob
                if vertex != 0:
                    parent[vertex] = source

                # Explore the vertex by learning policies for each outgoing edge
                for edge in self.abstract_graph[vertex]:

                    if explored[edge.target]:
                        nn_policies[vertex].append(None)
                    else:
                        # Set initial state distribution for edge
                        if vertex == 0:
                            start_dist = None
                        else:
                            start_dist = FiniteDistribution(reach_states[(source, vertex)])

                        # Learn policy
                        edge_policy, reach_env, log_info = edge.learn_policy(
                            env, hyperparams, vertex, start_dist, algo, res_model,
                            max_steps, safety_penalty, neg_inf, alpha, use_gpu, render)
                        nn_policies[vertex].append(edge_policy)

                        # update stats
                        num_edges_learned += 1
                        total_steps += log_info[-1][0]
                        total_time += log_info[-1][1]

                        # Compute reach probability and collect visited states
                        states_reached = []
                        reach_prob = 0
                        for _ in range(num_samples):
                            sarss = get_rollout(reach_env, edge_policy, False)
                            states = np.array([state for state, _, _, _ in sarss] + [sarss[-1][-1]])
                            total_steps += len(sarss)
                            if reach_env.cum_reward(states) > 0:
                                reach_prob += 1
                                if edge.target != vertex:
                                    states_reached.append(reach_env.get_state())
                        reach_prob = (reach_prob / num_samples)
                        print('\nReach Probability: {}'.format(reach_prob))
                        if edge.target != vertex:
                            if len(states_reached) > 0:
                                reach_states[(vertex, edge.target)] = states_reached
                                target_neg_log_prob = -np.log(reach_prob) + min_neg_log_prob[vertex]
                                heappush(queue, (target_neg_log_prob, edge.target, vertex))
                            else:
                                bad_edges.append((vertex, edge.target))
                        else:
                            success_measure = -np.log(reach_prob) + min_neg_log_prob[vertex]
                            success_measure = np.exp(-success_measure)
                            incomplete = False
                            print('Estimated Success Rate: {}'.format(success_measure))

                # Set the explored tag
                explored[vertex] = True
                if (vertex in self.final_vertices) and (success_measure > best_success):
                    best_success = success_measure
                    if success_measure > succ_thresh:
                        reached_final_vertex = True
                    u = vertex
                    abstract_policy[u] = u
                    while u != 0:
                        abstract_policy[parent[u]] = u
                        u = parent[u]

        # Change abstract policy to refer to edge number rather than vertex number
        for v in range(self.num_vertices):
            if abstract_policy[v] != -1:
                for i in range(len(self.abstract_graph[v])):
                    if self.abstract_graph[v][i].target == abstract_policy[v]:
                        abstract_policy[v] = i
                        break

        # Print bad edges
        if len(bad_edges) > 0:
            print('\nBad Edges:')
            for s, t in bad_edges:
                print('{} -> {}'.format(s, t))
            if incomplete:
                exit(1)

        return abstract_policy, nn_policies, [total_steps, total_time, num_edges_learned]

    def pretty_print(self):
        for i in range(self.num_vertices):
            targets = ''
            for edge in self.abstract_graph[i]:
                targets += ' ' + str(edge.target)
            print(str(i) + ' ->' + targets)
        print(self.final_vertices)


class ReachabilityEnv(gym.Env):
    '''
    Product of system and resource model.
    Terminates upon reaching a goal predicate (if specified).

    Parameters:
        env: gym.Env (with set_state() method)
        init_dist: Distribution (initial state distribution)
        final_pred: state, resource -> float (Goal of the reachability task)
        constraints: Constraints that need to be satisfied (defined reward function)
        res_model: Resource_Model (optional, can be None)
        max_steps: int
        safety_penalty: float (min penalty for violating constraints)
        neg_inf: float (negative reward for failing to satisfy constraints)
        alpha: float (alpha * original_reward will be added to reward)
    '''

    def __init__(self, env, init_dist=None, final_pred=None, constraints=[],
                 max_steps=100, res_model=None, safety_penalty=-1, neg_inf=-10,
                 alpha=0):
        self.wrapped_env = env
        self.init_dist = init_dist
        self.final_pred = final_pred
        self.constraints = constraints
        self.max_steps = max_steps
        self.safety_penalty = safety_penalty
        self.neg_inf = neg_inf
        self.alpha = alpha

        # extract dimensions from env
        self.orig_state_dim = self.wrapped_env.observation_space.shape[0]
        self.action_dim = self.wrapped_env.action_space.shape[0]

        # Dummy resource model
        if res_model is None:
            def delta(sys_state, res_state, sys_action):
                return np.array([])
            res_model = Resource_Model(self.orig_state_dim, self.action_dim, 0, np.array([]), delta)
        self.res_model = res_model

        obs_dim = self.orig_state_dim + self.res_model.res_init.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,))
        self.action_space = self.wrapped_env.action_space

        # Reset the environment
        self.reset()

    def reset(self):
        self.sys_state = self.wrapped_env.reset()
        if self.init_dist is not None:
            sim_state, self.res_state = self.init_dist.sample()
            self.sys_state = self.wrapped_env.set_sim_state(sim_state)
        else:
            self.res_state = self.res_model.res_init
        self.violated_constraints = 0
        self.prev_safety_reward = self.neg_inf
        self.t = 0
        return self.get_obs()

    def step(self, action):
        self.res_state = self.res_model.res_delta(self.sys_state, self.res_state, action)
        self.sys_state, r, _, _ = self.wrapped_env.step(action)
        self.t += 1

        reward = self.reward()
        reward = reward + self.alpha * min(r, 0)
        done = self.t > self.max_steps
        if (self.final_pred is not None) and (self.violated_constraints < len(self.constraints)):
            done = done or self.final_pred(self.sys_state, self.res_state) > 0

        return self.get_obs(), reward, done, {}

    def render(self):
        self.wrapped_env.render()
        print('System State: {} | Resource State: {}'.format(
            self.sys_state.tolist(), self.res_state.tolist()))

    def get_obs(self):
        return np.concatenate([self.sys_state, self.res_state])

    def get_state(self):
        return self.wrapped_env.get_sim_state(), self.res_state

    def reward(self):
        reach_reward = 0
        if self.final_pred is not None:
            reach_reward = self.final_pred(self.sys_state, self.res_state)

        safety_reward = self.prev_safety_reward
        set_new_vc = False
        for i in range(self.violated_constraints, len(self.constraints)):
            cur_constraint_val = self.constraints[i](self.sys_state, self.res_state)
            safety_reward = max(safety_reward, cur_constraint_val)
            if not set_new_vc:
                if cur_constraint_val <= 0:
                    self.violated_constraints += 1
                else:
                    set_new_vc = True
        safety_reward = min(safety_reward, 0)
        if safety_reward < 0:
            safety_reward = min(safety_reward, self.safety_penalty)
            self.prev_safety_reward = safety_reward

        return reach_reward + safety_reward

    def cum_reward(self, states):
        reach_reward = self.neg_inf
        safety_reward = -self.neg_inf
        violated_constraints = 0
        for s in states:
            sys_state = s[:self.orig_state_dim]
            res_state = s[self.orig_state_dim:]
            if self.final_pred is not None:
                reach_reward = max(reach_reward, self.final_pred(sys_state, res_state))

            cur_safety_reward = self.neg_inf
            for i in range(violated_constraints, len(self.constraints)):
                tmp_reward = self.constraints[i](sys_state, res_state)
                if tmp_reward <= 0:
                    violated_constraints += 1
                else:
                    cur_safety_reward = tmp_reward
                    break
            safety_reward = min(safety_reward, cur_safety_reward)
        if self.final_pred is None:
            reach_reward = -self.neg_inf
        return min(reach_reward, safety_reward)

    def close_viewer(self):
        self.wrapped_env.close()


class ConstrainedEnv(ReachabilityEnv):
    '''
    Environment for the full tasks enforcing constraints on the chosen abstract path.

    Parameters:
        env: gym.Env (with set_state() method)
        init_dist: Distribution (initial state distribution)
        abstract_reach: AbstractReachability
        abstract_policy: list of int (edge to choose in each abstract state)
        res_model: Resource_Model (optional, can be None)
        max_steps: int
    '''

    def __init__(self, env, abstract_reach, abstract_policy,
                 res_model=None, max_steps=100):
        self.abstract_graph = abstract_reach.abstract_graph
        self.final_vertices = abstract_reach.final_vertices
        self.abstract_policy = abstract_policy
        super(ConstrainedEnv, self).__init__(env, max_steps=max_steps,
                                             res_model=res_model)

    def reset(self):
        obs = super(ConstrainedEnv, self).reset()
        self.vertex = 0
        self.edge = self.abstract_graph[0][self.abstract_policy[0]]
        self.blocked_constraints = 0
        self.update_blocked_constraints()
        return obs

    def step(self, action):
        obs, _, done, info = super(ConstrainedEnv, self).step(action)
        if self.blocked_constraints >= len(self.edge.constraints):
            return obs, 0, True, info

        if self.edge.predicate is not None:
            if self.edge.predicate(self.sys_state, self.res_state) > 0:
                self.vertex = self.edge.target
                self.edge = self.abstract_graph[self.vertex][self.abstract_policy[self.vertex]]
                self.blocked_constraints = 0

        self.update_blocked_constraints()
        if self.blocked_constraints >= len(self.edge.constraints):
            return obs, 0, True, info

        reward = 0
        if done and self.vertex in self.final_vertices:
            reward = 1
        return obs, reward, done, info

    def update_blocked_constraints(self):
        for i in range(self.blocked_constraints, len(self.edge.constraints)):
            if self.edge.constraints[i](self.sys_state, self.res_state) > 0:
                break
            self.blocked_constraints += 1


class HierarchicalPolicy:

    def __init__(self, abstract_policy, nn_policies, abstract_graph, sys_dim):
        self.abstract_policy = abstract_policy
        self.nn_policies = nn_policies
        self.abstract_graph = abstract_graph
        self.vertex = 0
        self.edge = self.abstract_graph[0][self.abstract_policy[0]]
        self.sys_dim = sys_dim

    def get_action(self, state):
        sys_state = state[:self.sys_dim]
        res_state = state[self.sys_dim:]
        if self.edge.predicate is not None:
            if self.edge.predicate(sys_state, res_state) > 0:
                self.vertex = self.edge.target
                self.edge = self.abstract_graph[self.vertex][self.abstract_policy[self.vertex]]
        return self.nn_policies[self.vertex][self.abstract_policy[self.vertex]].get_action(state)

    def reset(self):
        self.vertex = 0
        self.edge = self.abstract_graph[0][self.abstract_policy[0]]
