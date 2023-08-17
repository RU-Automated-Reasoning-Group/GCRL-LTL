from spectrl.main.monitor import Resource_Model
from spectrl.main.monitor import Compiled_Spec
from spectrl.rl.ars_discrete import ARSParams, NNParams, NNPolicy, NNPolicySimple, ars
from spectrl.util.rl import get_rollout

import numpy as np
import gym


class ProductMDP:
    '''
    Parameters:
        system : System MDP (no need for reward function)
        action_dim: action space dimension for the system
        res_model : Resource_Model (optional)
        spec : TaskSpec
        min_reward (C_l) = Min possible unshaped reward
        local_reward_bound (C_u) = Max possible absolute value of local reward (quant. sem. value)
    '''

    def __init__(self, system, action_dim, spec, min_reward, local_reward_bound,
                 res_model=None, use_shaped_rewards=True):
        self.system = system
        init_system_state = self.system.reset()
        system_state_dim = len(init_system_state)
        if res_model is None:
            def delta(sys_state, res_state, sys_action):
                return np.array([])
            res_model = Resource_Model(
                system_state_dim, action_dim, 0, np.array([]), delta)
        monitor = spec.get_monitor(
            system_state_dim, res_model.res_dim, local_reward_bound)
        self.spec = Compiled_Spec(
            res_model, monitor, min_reward, local_reward_bound)
        self.state = (np.append(init_system_state,
                                self.spec.init_extra_state()), 0)
        self.is_shaped = use_shaped_rewards

    def reset(self):
        self.state = (np.append(self.system.reset(),
                                self.spec.init_extra_state()), 0)
        return self.state

    def step(self, action):
        next_state, rew, done, render = self.system.step(
            action[:self.spec.action_dim])
        res_reg_state, monitor_state = self.spec.extra_step(self.state, action)
        self.state = (np.append(next_state, res_reg_state), monitor_state)
        return self.state, rew, done, render

    def action_dim(self):
        return self.spec.total_action_dim

    def state_dim(self):
        return self.spec.total_state_dim

    def cum_reward(self, rollout):
        if self.is_shaped:
            return self.spec.cum_reward_shaped(rollout)
        else:
            return self.spec.cum_reward_unshaped(rollout)

    def render(self):
        self.system.render()


class QSRewardEnv(gym.Env):

    def __init__(self, env, spec, res_model=None, fast_semantics=False):
        self.wrapped_env = env
        self.spec = spec
        self.fast_semantics = fast_semantics
        self.action_space = self.wrapped_env.action_space
        self.sys_dim = self.wrapped_env.observation_space.shape[0]

        # Dummy resource model
        if res_model is None:
            action_dim = self.action_space.shape[0]

            def delta(sys_state, res_state, sys_action):
                return np.array([])
            res_model = Resource_Model(self.sys_dim, action_dim, 0, np.array([]), delta)
        self.res_model = res_model

        total_dim = self.sys_dim + self.res_model.res_init.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(total_dim,))

    def reset(self):
        self.resource_state = self.res_model.res_init
        self.system_state = self.wrapped_env.reset()
        return self.get_obs()

    def step(self, action):
        self.resource_state = self.res_model.res_delta(
            self.system_state, self.resource_state, action)
        self.system_state, reward, done, info = self.wrapped_env.step(action)
        return self.get_obs(), reward, done, info

    def render(self):
        print(self.system_state.tolist())
        return self.wrapped_env.render()

    def cum_reward(self, traj):
        if self.fast_semantics:
            return self.spec.quantitative_semantics_fast(traj, self.sys_dim)
        else:
            return self.spec.quantitative_semantics(traj, self.sys_dim)

    def get_obs(self):
        return np.concatenate([self.system_state, self.resource_state])


class HyperParams:
    '''
    HyperParameters:
        hidden_layer_dim = Number of neurons in the hidden layers
        actions confined to the range [-action_bound, action_bound]
        n_iters: int (ending condition)
        n_samples: int (N)
        n_top_samples: int (b)
        delta_std (nu)
        lr: float (alpha)
        min_lr: float (minimum alpha)
    '''

    def __init__(self, hidden_layer_dim, action_bound, n_iters, n_samples, n_top_samples, delta_std,
                 lr, min_lr, log_interval=1):
        self.hidden_dim = hidden_layer_dim
        self.action_bound = action_bound
        self.ars_params = ARSParams(
            n_iters, n_samples, n_top_samples, delta_std, lr, min_lr, log_interval)


def learn_policy(env, params, variant='normal'):
    '''
    Learn Policy for Product MDP using ARS.

    Parameters:
        env : ProductMDP
        params : HyperParams
        variant : {'normal', 'unshaped', 'state_based', 'deterministic'}
            'normal' - SPECTRL
            'unshaped' - Use unshaped rewards
            'state-based' - State based policy that does not depend on monitor state
            'deterministic' - Use greedy deterministic monitor
    '''

    # dimension of input vector to nn
    nn_input_dim = env.spec.total_state_dim
    if variant == 'state-based':
        nn_input_dim = env.spec.register_split

    # dimension of output vector of nn
    nn_output_dim = env.spec.total_action_dim
    if variant == 'deterministic':
        nn_output_dim = env.spec.action_dim

    policy_params = NNParams(nn_input_dim, nn_output_dim, params.action_bound,
                             params.hidden_dim, env.spec.monitor.n_states)

    policy = NNPolicy(policy_params)
    if variant == 'state-based':
        policy = NNPolicySimple(policy_params)

    log_info = ars(env, policy, params.ars_params)

    return policy, log_info


def print_rollout(env, policy, render=False):
    '''
    Print formatted rollout.
    '''
    test_rollout = get_rollout(env, policy, render)
    state_list = [(state[0][:env.spec.state_dim].tolist(),
                   state[0][env.spec.state_dim:env.spec.register_split].tolist(),
                   state[1],
                   state[0][env.spec.register_split:].tolist())
                  for state, _, _, _ in test_rollout]
    print('**** Rollout ****')
    for state in state_list:
        print(str(["{0:0.2f}".format(i) for i in state[0]]).replace("'", "") + ", "
              + str(["{0:0.2f}".format(i)
                     for i in state[1]]).replace("'", "") + ", "
              + str(state[2]) + ", "
              + str(["{0:0.2f}".format(i) for i in state[3]]).replace("'", ""))


class FinalStateWrapper:
    '''
    Env wrapper to add dummy transition in the end.
    This is needed in some cases beacuse the monitoring is only for rollout[:-1].
    '''

    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self._done_state = None

    def reset(self):
        self._done_state = None
        return self._wrapped_env.reset()

    def step(self, action):
        if self._done_state is None:
            state, rew, done, info = self._wrapped_env.step(action)
            if done:
                self._done_state = state
            return state, rew, False, info
        else:
            return self._done_state, 0, True, None

    def render(self):
        self._wrapped_env.render()
