import numpy as np


class Resource_Model:
    '''
    Model for resources.

    Parameters:
        state_dim : int
        action_dim : int
        res_dim : int
        res_init : np.array(res_dim)
        res_delta : np.array(state_dim), np.array(res_dim), np.array(action_dim)
                    -> np.array(res_dim)
    '''

    def __init__(self, state_dim, action_dim, res_dim, res_init, res_delta):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.res_dim = res_dim
        self.res_init = res_init
        self.res_delta = res_delta


class Monitor_Automaton:
    '''
    Class for representing monitors

    Parameters:
        n_states : int
        n_registers : int
        input_dim : int (state_dim + res_dim)
        init_registers : np.array(n_registers)
        transitions : adjacency list of transitions of the form (q,p,u) where,
                    q : int (monitor state)
                    p : np.array(input_dim) , np.array(n_registers) -> (Bool,Float {quant. sym.})
                    u : np.array(input_dim) , np.array(n_registers) -> np.array(n_registers)
        rewards : list of n_states reward functions for final states (others are None)
                rewards[i] : np.array(input_dim) , np.array(n_registers) -> Float
    '''

    def __init__(self, n_states, n_registers, input_dim, init_registers, transitions, rewards):

        self.n_states = n_states
        self.n_registers = n_registers
        self.input_dim = input_dim
        self.init_registers = init_registers
        self.transitions = transitions
        self.rewards = rewards

# ==================================================================================================
# Useful functions on monitors


def find_state_depths(monitor):
    depths = [0] * monitor.n_states
    incoming = [0] * monitor.n_states

    for q1 in range(monitor.n_states):
        for (q2, _, _) in monitor.transitions[q1]:
            if q1 != q2:
                incoming[q2] += 1

    topo_sort = []
    for q in range(monitor.n_states):
        if incoming[q] == 0:
            topo_sort.append(q)
            depths[q] = 0

    for i in range(monitor.n_states):
        for (q, _, _) in monitor.transitions[topo_sort[i]]:
            if q != topo_sort[i]:
                incoming[q] -= 1
                depths[q] = max(depths[q], depths[topo_sort[i]]+1)
                if incoming[q] == 0:
                    topo_sort.append(q)

    return depths


class Compiled_Spec:
    '''
    Compiled Specification consisting of resource model and reward monitor.

    Parameters:
        resource : Resource_Model
        monitor : Monitor_Automaton
        min_reward : float
        local_reward_bound : float (C)
    '''

    def __init__(self, resource, monitor, min_reward, local_reward_bound):

        self.state_dim = resource.state_dim
        self.action_dim = resource.action_dim

        # Resource
        self.resource = resource

        # Monitor
        self.monitor = monitor
        self.extra_action_dim = max(map(len, self.monitor.transitions))
        self.depths = find_state_depths(self.monitor)
        self.max_depth = max(self.depths)

        # Product MDP
        self.total_state_dim = self.state_dim + \
            self.resource.res_dim + self.monitor.n_registers
        self.total_action_dim = self.action_dim + self.extra_action_dim
        self.min_reward = min_reward
        self.local_reward_bound = local_reward_bound

        self.register_split = self.state_dim + self.resource.res_dim

    def extract_state_components(self, state):
        '''
        Extract state components.
        state : (np.array(total_state_dim), int)
        '''
        sys_state = state[0][:self.state_dim]
        res_state = state[0][self.state_dim:self.register_split]
        register_state = state[0][self.register_split:]
        monitor_state = state[1]

        return sys_state, res_state, register_state, monitor_state

    def extra_step(self, state, action):
        '''
        Step function for Resource, Monitor state and Registers.
        state : (np.array(total_state_dim), int)
        action : np.array(total_action_dim) or np.array(action_dim)
        return value : (np.array(resource.res_dim + monitor.n_registers), int)
        '''
        sys_state, res_state, register_state, monitor_state = self.extract_state_components(
            state)

        sys_action = action[:self.action_dim]

        # Step the Resource state
        new_res_state = np.array([])
        if self.resource.res_dim > 0:
            new_res_state = self.resource.res_delta(
                sys_state, res_state, sys_action)

        # Step the Monitor and Register state
        # Assumes at-least one predicate of the outgoing edges will be satisfied
        # start with an edge and explore to find the edge that is feasible
        edge = 0

        if len(action) < self.total_action_dim:
            q, _, _ = self.monitor.transitions[monitor_state][0]
            if q == monitor_state:
                edge = 1
            else:
                edge = 0
        else:
            mon_action = action[self.action_dim:]
            edge = np.argmax(
                mon_action[:len(self.monitor.transitions[monitor_state])])

        outgoing_transitions = self.monitor.transitions[monitor_state]
        sys_res_state = np.concatenate([sys_state, res_state])
        for i in range(len(outgoing_transitions)):
            (q, p, u) = outgoing_transitions[(
                i+edge) % len(outgoing_transitions)]
            satisfied, _ = p(sys_res_state, register_state)
            if satisfied:
                new_register_state = u(sys_res_state, register_state)
                return (np.concatenate([new_res_state, new_register_state]), q)

    def init_extra_state(self):
        '''
        Initial values of the extra state space.
        return value : np.array(resource.res_dim + monitor.n_registers)
        '''
        return np.concatenate([self.resource.res_init, self.monitor.init_registers])

    def shaped_reward(self, sys_res_state, register_state, monitor_state):
        '''
        Shaped reward for a state.
        sys_res_state : np.array(state_dim + resource.res_dim)
        monitor_state : int
        register_state : np.array(monitor.n_registers)
        '''
        rew = -10000000
        for (q, p, u) in self.monitor.transitions[monitor_state]:
            if q == monitor_state:
                continue
            _, edge_rew = p(sys_res_state, register_state)
            rew = max(rew, edge_rew)
        return self.min_reward \
            + rew \
            + (self.local_reward_bound) * \
            (self.depths[monitor_state] - self.max_depth)

    def cum_reward_shaped(self, rollout):
        '''
        Cumulative reward for a rollout (shaped).
        rollout : [(np.array, int)]
        '''
        last_state = rollout[len(rollout)-1]

        # Final State
        if self.monitor.rewards[last_state[1]] is not None:
            (last_sys_state, last_res_state, last_register_state,
             last_monitor_state) = self.extract_state_components(last_state)
            last_sys_res_state = np.concatenate(
                [last_sys_state, last_res_state])
            return self.monitor.rewards[last_monitor_state](last_sys_res_state, last_register_state)

        # Non-final state
        rew = -10000000
        for state in rollout:
            if(state[1] == last_state[1]):
                (sys_state, res_state, register_state,
                 monitor_state) = self.extract_state_components(state)
                sys_res_state = np.concatenate([sys_state, res_state])
                rew = max(rew, self.shaped_reward(
                    sys_res_state, register_state, monitor_state))

        return rew

    def cum_reward_unshaped(self, rollout):
        '''
        Cumulative reward for a rollout (unshaped).
        rollout : [(np.array, int)]
        '''
        last_state = rollout[len(rollout)-1]

        # Final State
        if self.monitor.rewards[last_state[1]] is not None:
            (last_sys_state, last_res_state, last_register_state,
             last_monitor_state) = self.extract_state_components(last_state)
            last_sys_res_state = np.concatenate(
                [last_sys_state, last_res_state])
            return self.monitor.rewards[last_monitor_state](last_sys_res_state, last_register_state)

        # Non-final state
        return self.min_reward - self.local_reward_bound
