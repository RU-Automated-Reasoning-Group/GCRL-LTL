from spectrl.util.rl import get_rollout, test_policy, discounted_reward

import time
import torch
import numpy as np


# Parameters for training a policy neural net.
#
# state_dim: int (n)
# action_dim: int (p)
# hidden_dim: int
# dir: str
# fname: str
class NNParams:
    def __init__(self, env, hidden_dim):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high
        self.hidden_dim = hidden_dim


# Parameters for augmented random search policy.
#
# n_iters: int (ending condition)
# n_samples: int (N)
# n_top_samples: int (b)
# delta_std (nu)
# lr: float (alpha)
class ARSParams:
    def __init__(self, n_iters, n_samples, n_top_samples, delta_std, lr, min_lr, log_interval=1):
        self.n_iters = n_iters
        self.n_samples = n_samples
        self.n_top_samples = n_top_samples
        self.delta_std = delta_std
        self.lr = lr
        self.min_lr = min_lr
        self.log_interval = log_interval


# HyperParameters:
#     hidden_layer_dim = Number of neurons in the hidden layers
#     actions confined to the range [-action_bound, action_bound]
#     n_iters: int (ending condition)
#     n_samples: int (N)
#     n_top_samples: int (b)
#     delta_std (nu)
#     lr: float (alpha)
#     min_lr: float (minimum alpha)
class HyperParams:
    def __init__(self, hidden_layer_dim, n_iters, n_samples, n_top_samples, delta_std, lr, min_lr):
        self.hidden_dim = hidden_layer_dim
        self.ars_params = ARSParams(n_iters, n_samples, n_top_samples, delta_std, lr, min_lr)


# Neural network policy.
class NNPolicy:
    # Initialize the neural network.
    #
    # params: NNParams
    def __init__(self, params):
        # Step 1: Parameters
        self.params = params

        # Step 2: Construct neural network

        # Step 2a: Construct the input layer
        self.input_layer = torch.nn.Linear(self.params.state_dim, self.params.hidden_dim)

        # Step 2b: Construct the hidden layer
        self.hidden_layer = torch.nn.Linear(self.params.hidden_dim, self.params.hidden_dim)

        # Step 2c: Construct the output layer
        self.output_layer = torch.nn.Linear(self.params.hidden_dim, self.params.action_dim)

        # Step 3: Construct input normalization
        self.mu = np.zeros(self.params.state_dim)
        self.sigma_inv = np.ones(self.params.state_dim)

    # Get the action to take in the current state.
    #
    # state: np.array([state_dim])
    def get_action(self, state):
        # Step 1: Normalize state
        state = (state - self.mu) * self.sigma_inv

        # Step 2: Convert to torch
        state = torch.tensor(state, dtype=torch.float)

        # Step 3: Apply the input layer
        hidden = torch.relu(self.input_layer(state))

        # Step 4: Apply the hidden layer
        hidden = torch.relu(self.hidden_layer(hidden))

        # Step 5: Apply the output layer
        output = torch.tanh(self.output_layer(hidden))

        # Step 6: Convert to numpy
        actions = output.detach().numpy()

        # Step 7: Scale the outputs
        actions = self.params.action_bound * actions

        return actions

    # Get the best action to take in the current state.
    #
    # state: np.array([state_dim])
    def get_best_action(self, state):
        return self.get_action(state)

    # Construct the set of parameters for the policy.
    #
    # nn_policy: NNPolicy
    # return: list of torch parameters
    def parameters(self):
        parameters = []
        parameters.extend(self.input_layer.parameters())
        parameters.extend(self.hidden_layer.parameters())
        parameters.extend(self.output_layer.parameters())
        return parameters


# Run augmented random search.
#
# env: Environment
# nn_policy: NNPolicy
# params: ARSParams
def ars(env, nn_policy, params, cum_reward=False):
    # Step 1: Save original policy
    nn_policy_orig = nn_policy
    best_policy = nn_policy
    best_reward = -1e9

    # Logging information
    log_info = []
    num_steps = 0
    start_time = time.time()

    # Step 2: Initialize state distribution estimates
    mu_sum = np.zeros(nn_policy.params.state_dim)
    sigma_sq_sum = np.ones(nn_policy.params.state_dim) * 1e-5
    n_states = 0

    # Step 3: Training iterations
    for i in range(params.n_iters):
        # Step 3a: Sample deltas
        deltas = []
        for _ in range(params.n_samples):
            # i) Sample delta
            delta = _sample_delta(nn_policy)

            # ii) Construct perturbed policies
            nn_policy_plus = _get_delta_policy(nn_policy, delta, params.delta_std)
            nn_policy_minus = _get_delta_policy(nn_policy, delta, -params.delta_std)

            # iii) Get rollouts
            sarss_plus = get_rollout(env, nn_policy_plus, False)
            sarss_minus = get_rollout(env, nn_policy_minus, False)
            num_steps += (len(sarss_plus) + len(sarss_minus))

            # iv) Estimate cumulative rewards
            if not cum_reward:
                r_plus = discounted_reward(sarss_plus, 1)
                r_minus = discounted_reward(sarss_minus, 1)
            else:
                r_plus = env.cum_reward([s for s, _, _, _ in sarss_plus] + [sarss_plus[-1][-1]])
                r_minus = env.cum_reward([s for s, _, _, _ in sarss_minus] + [sarss_minus[-1][-1]])

            # v) Save delta
            deltas.append((delta, r_plus, r_minus))

            # v) Update estimates of normalization parameters
            states = np.array([state for state, _, _, _ in sarss_plus + sarss_minus])
            mu_sum += np.sum(states)
            sigma_sq_sum += np.sum(np.square(states))
            n_states += len(states)

        # Step 3b: Sort deltas
        deltas.sort(key=lambda delta: -max(delta[1], delta[2]))

        # Step 3c: Compute the sum of the deltas weighted by their reward differences
        delta_sum = [torch.zeros(delta_cur.shape) for delta_cur in deltas[0][0]]
        for j in range(params.n_top_samples):
            # i) Unpack values
            delta, r_plus, r_minus = deltas[j]

            # ii) Add delta to the sum
            for k in range(len(delta_sum)):
                delta_sum[k] += (r_plus - r_minus) * delta[k]

        # Step 3d: Compute standard deviation of rewards
        sigma_r = np.std([delta[1] for delta in deltas] + [delta[2] for delta in deltas])

        # Step 3e: Compute step length
        delta_step = [(params.lr * params.delta_std / (params.n_top_samples * sigma_r + 1e-8))
                      * delta_sum_cur for delta_sum_cur in delta_sum]

        # Step 3f: Update policy weights
        nn_policy = _get_delta_policy(nn_policy, delta_step, 1.0)

        # Step 3g: Update normalization parameters
        nn_policy.mu = mu_sum / n_states
        nn_policy.sigma_inv = 1.0 / np.sqrt(sigma_sq_sum / n_states)

        # Step 3h: Logging
        if i % params.log_interval == 0:
            avg_reward, succ_rate = test_policy(env, nn_policy, 100, use_cum_reward=cum_reward)
            time_taken = (time.time() - start_time)/60
            print('\nSteps taken at iteration {}: {}'.format(i, num_steps))
            print('Time taken at iteration {}: {} mins'.format(i, time_taken))
            print('Expected reward at iteration {}: {}'.format(i, avg_reward))
            if cum_reward:
                print('Estimated success rate at iteration {}: {}'.format(i, succ_rate))
                log_info.append([num_steps, time_taken, avg_reward, succ_rate])
            else:
                log_info.append([num_steps, time_taken, avg_reward])

            # Step 4: Copy new weights and normalization parameters to original policy
            if avg_reward >= best_reward:
                best_reward = avg_reward
                best_policy = nn_policy

    nn_policy = best_policy

    for param, param_orig in zip(nn_policy.parameters(), nn_policy_orig.parameters()):
        param_orig.data.copy_(param.data)
    nn_policy_orig.mu = nn_policy.mu
    nn_policy_orig.sigma_inv = nn_policy.sigma_inv

    return log_info


# Construct random perturbations to neural network parameters.
#
# nn_policy: NNPolicy
# return: [torch.tensor] (list of torch tensors that is the same shape as nn_policy.parameters())
def _sample_delta(nn_policy):
    delta = []
    for param in nn_policy.parameters():
        delta.append(torch.normal(torch.zeros(param.shape, dtype=torch.float)))
    return delta


# Construct the policy perturbed by the given delta
#
# nn_policy: NNPolicy
# delta: [torch.tensor] (list of torch tensors that is the same shape as nn_policy.parameters())
# sign: float (should be 1.0 or -1.0, for convenience)
# return: NNPolicy
def _get_delta_policy(nn_policy, delta, sign):
    # Step 1: Construct the perturbed policy
    nn_policy_delta = NNPolicy(nn_policy.params)

    # Step 2: Set normalization of the perturbed policy
    nn_policy_delta.mu = nn_policy.mu
    nn_policy_delta.sigma_inv = nn_policy.sigma_inv

    # Step 3: Set the weights of the perturbed policy
    for param, param_delta, delta_cur in zip(nn_policy.parameters(), nn_policy_delta.parameters(),
                                             delta):
        param_delta.data.copy_(param.data + sign * delta_cur)

    return nn_policy_delta
