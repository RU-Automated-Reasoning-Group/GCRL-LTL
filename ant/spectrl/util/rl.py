import numpy as np
import gym


class RandomPolicy:
    '''
    Policy that takes random actions

    Parameters:
        action_dim : int
        action_bound : float (bound on absolute value of each component)
    '''

    def __init__(self, action_dim, action_bound):
        self.action_dim = action_dim
        self.action_bound = action_bound

    def get_action(self, state):
        return ((np.random.random_sample((self.action_dim,)) * self.action_bound * 2)
                - self.action_bound)


class ObservationWrapper(gym.Env):
    '''
    Wraps an environment modifying dictionary obervations into array observations.
    '''

    def __init__(self, env, keys, relative=None, max_timesteps=10000):
        self.env = env
        self.keys = keys
        self.relative = relative
        self.max_timesteps = max_timesteps

        obs = self.env.reset()
        obs_dim = sum([obs[key].shape[0] for key in self.keys])
        if self.relative is not None:
            obs_dim += relative[0][2] - relative[0][1]
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,))

    def reset(self):
        self.t = 0
        obs = self.env.reset()
        return self.flatten_obs(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.t += 1
        done = done or self.t > self.max_timesteps
        return self.flatten_obs(obs), rew, done, info

    def render(self):
        return self.env.render()

    def get_sim_state(self):
        return self.env.get_sim_state()

    def set_sim_state(self, state):
        return self.flatten_obs(self.env.set_sim_state(state))

    def close(self):
        self.env.close()

    def flatten_obs(self, obs):
        flat_obs = np.concatenate([obs[key] for key in self.keys])
        if self.relative is not None:
            (key1, i1, j1) = self.relative[0]
            (key2, i2, j2) = self.relative[1]
            rel_obs = obs[key1][i1:j1] - obs[key2][i2:j2]
            flat_obs = np.concatenate([flat_obs, rel_obs])
        return flat_obs


def get_rollout(env, policy, render, max_timesteps=10000, stateful_policy=False):
    '''
    Compute a single rollout.

    Parameters:
        env: gym.Env
        policy: Object (implements function get_action: state -> action)
        render: bool

    Returns: [(np.array, np.array, float, np.array)]
             ((state, action, reward, next_state) tuples)
    '''
    # Step 1: Initialization
    state = env.reset()
    if stateful_policy:
        policy.reset()
    done = False

    # Step 2: Compute rollout
    sarss = []
    steps = 0
    while (not done) and (steps < max_timesteps):
        # Step 2a: Render environment
        if render:
            env.render()

        # Step 2b: Action
        action = policy.get_action(state)

        # Step 2c: Transition environment
        next_state, reward, done, _ = env.step(action)

        # Step 2d: Rollout (s, a, r)
        sarss.append((state, action, reward, next_state))

        # Step 2e: Update state
        state = next_state
        steps += 1

    # Step 3: Render final state
    if render:
        env.render()

    return sarss


def discounted_reward(sarss, gamma):
    sarss_rev = sarss.copy()
    sarss_rev.reverse()
    reward = 0.0
    for _, _, r, _ in sarss_rev:
        reward = r + gamma*reward
    return reward


def test_policy(env, policy, n_rollouts, gamma=1, use_cum_reward=False,
                get_steps=False, max_timesteps=10000, stateful_policy=False):
    '''
    Estimate the cumulative reward of the policy.

    Parameters:
        env: gym.Env
        policy: Object (implements function get_action: state -> action)
        n_rollouts: int

    Returns:
        avg_reward, avg_prob if get_steps is False
        avg_reward, avg_prob, num_steps if get_steps is True
    '''
    cum_reward = 0.0
    succ_rate = 0.0
    num_steps = 0
    for _ in range(n_rollouts):
        sarss = get_rollout(env, policy, False, max_timesteps=max_timesteps,
                            stateful_policy=stateful_policy)
        num_steps += len(sarss)
        if use_cum_reward:
            tmp_reward = env.cum_reward(
                np.array([state for state, _, _, _ in sarss] + [sarss[-1][-1]]))
        else:
            tmp_reward = discounted_reward(sarss, gamma)
        cum_reward += tmp_reward
        if tmp_reward > 0:
            succ_rate += 1.0
    if get_steps:
        return cum_reward / n_rollouts, succ_rate / n_rollouts, num_steps
    return cum_reward / n_rollouts, succ_rate / n_rollouts


def print_performance(environment, policy, gamma=1, use_cum_reward=False,
                      n_rollouts=100, max_timesteps=10000, stateful_policy=False):
    reward, prob = test_policy(environment, policy, n_rollouts, gamma=gamma,
                               use_cum_reward=use_cum_reward, max_timesteps=max_timesteps,
                               stateful_policy=stateful_policy)
    print('\n' + '-'*40)
    print('Estimated Reward: {}'.format(reward))
    print('Estimated Reaching Probability: {}'.format(prob))
    print('-'*40 + '\n')
    return reward, prob


def print_rollout(env, policy, state_dim=-1, stateful_policy=False):
    sarss = get_rollout(env, policy, False, stateful_policy=stateful_policy)
    for s, _, _, _ in sarss:
        print(s.tolist()[:state_dim])
    print(sarss[-1][-1].tolist()[:state_dim])
