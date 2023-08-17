import torch
import numpy as np
import time

from spectrl.util.rl import get_rollout, test_policy


class NNParams:
    '''
    Defines the neural network architecture.

    Parameters:
        state_dim: int (continuous state dimension for nn input)
        action_dim: int (action space dimension for nn output)
        hidden_dim: int (hidden states in the nn)
        action_bound: float
        num_discrete_states: int (number of different discrete states possible)
    '''

    def __init__(self, state_dim, action_dim, action_bound, hidden_dim, num_discrete_states):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.hidden_dim = hidden_dim
        self.num_discrete_states = num_discrete_states


class ARSParams:
    '''
    HyperParameters for augmented random search.

    Parameters:
        n_iters: int (ending condition)
        n_samples: int (N)
        n_top_samples: int (b)
        delta_std (nu)
        lr: float (alpha)
        min_lr: float (minimum alpha)
    '''

    def __init__(self, n_iters, n_samples, n_top_samples, delta_std, lr, min_lr, log_interval=1):
        self.n_iters = n_iters
        self.n_samples = n_samples
        self.n_top_samples = n_top_samples
        self.delta_std = delta_std
        self.lr = lr
        self.min_lr = min_lr
        self.log_interval = log_interval


class NNPolicy:
    '''
    Neural network policy.
    params: NNParams
    '''

    def __init__(self, params):
        # Step 1: Parameters
        self.params = params

        # Step 2: Construct num_discrete_states neural networks
        self.input_layers = []
        self.hidden_layers = []
        self.output_layers = []

        for i in range(self.params.num_discrete_states):

            # Step 2a: Construct the input layer
            input_layer = torch.nn.Linear(
                self.params.state_dim, self.params.hidden_dim)

            # Step 2b: Construct the hidden layer
            hidden_layer = torch.nn.Linear(
                self.params.hidden_dim, self.params.hidden_dim)

            # Step 2c: Construct the output layer
            output_layer = torch.nn.Linear(
                self.params.hidden_dim, self.params.action_dim)

            self.input_layers.append(input_layer)
            self.hidden_layers.append(hidden_layer)
            self.output_layers.append(output_layer)

        # Step 3: Construct input normalization
        self.mu = np.zeros(self.params.state_dim)
        self.sigma_inv = np.ones(self.params.state_dim)

        # Set requires_grad to False
        for param in self.parameters():
            param.requires_grad_(False)

    def get_input(self, state):
        '''
        Get the neural network input from the full state
        state is a pair (continuous state, discrete state).
        '''
        return state[0][:self.params.state_dim]

    def get_action(self, state):
        '''
        Get the action to take in the current state.
        state: (np.array, int)
        '''
        # Step 0: Separate discrete and continuous components
        input = self.get_input(state)

        # Step 1: Normalize state
        input = (input - self.mu) * self.sigma_inv

        # Step 2: Convert to torch
        input = torch.tensor(input, dtype=torch.float)

        # Step 3: Apply the input layer
        hidden = torch.relu(self.input_layers[state[1]](input))

        # Step 4: Apply the hidden layer
        hidden = torch.relu(self.hidden_layers[state[1]](hidden))

        # Step 5: Apply the output layer
        output = torch.tanh(self.output_layers[state[1]](hidden))

        # Step 6: Convert to numpy
        actions = output.detach().numpy()

        return self.params.action_bound * actions

    def parameters(self):
        '''
        Construct the set of parameters for the policy.
        Returns a list of torch parameters.
        '''
        parameters = []
        for i in range(self.params.num_discrete_states):
            parameters.extend(self.input_layers[i].parameters())
            parameters.extend(self.hidden_layers[i].parameters())
            parameters.extend(self.output_layers[i].parameters())
        return parameters


class NNPolicySimple:
    '''
    Neural network policy that only looks at system state.
    Ignores discrete state.
    Only looks at first state_dim components of continuous state.
    params: NNParams
    '''

    def __init__(self, params):
        # Step 1: Parameters
        self.params = params

        # Step 2a: Construct the input layer
        self.input_layer = torch.nn.Linear(
            self.params.state_dim, self.params.hidden_dim)

        # Step 2b: Construct the hidden layer
        self.hidden_layer = torch.nn.Linear(
            self.params.hidden_dim, self.params.hidden_dim)

        # Step 2c: Construct the output layer
        self.output_layer = torch.nn.Linear(
            self.params.hidden_dim, self.params.action_dim)

        # Step 3: Construct input normalization
        self.mu = np.zeros(self.params.state_dim)
        self.sigma_inv = np.ones(self.params.state_dim)

    def get_input(self, state):
        return state[0][:self.params.state_dim]

    def get_action(self, state):
        '''
        Get the action to take in the current state.
        state: (np.array, int)
        '''
        # Step 0: Extract the system state
        input = self.get_input(state)

        # Step 1: Normalize state
        input = (input - self.mu) * self.sigma_inv

        # Step 2: Convert to torch
        input = torch.tensor(input, dtype=torch.float)

        # Step 3: Apply the input layer
        hidden = torch.relu(self.input_layer(input))

        # Step 4: Apply the hidden layer
        hidden = torch.relu(self.hidden_layer(hidden))

        # Step 5: Apply the output layer
        output = torch.tanh(self.output_layer(hidden))

        # Step 6: Convert to numpy
        actions = output.detach().numpy()

        return self.params.action_bound * actions

    def parameters(self):
        '''
        Construct the set of parameters for the policy.
        Returns a list of torch parameters.
        '''
        parameters = []
        parameters.extend(self.input_layer.parameters())
        parameters.extend(self.hidden_layer.parameters())
        parameters.extend(self.output_layer.parameters())
        return parameters


def ars(env, nn_policy, params):
    '''
    Run augmented random search.

    Parameters:
        env: gym.Env (state is expected to be a pair (np.array, int))
            Also expected to provide cum_reward() function.
        nn_policy: NNPolicy
        params: ARSParams
    '''

    best_policy = nn_policy
    best_success_rate = 0
    best_reward = -1e9
    log_info = []
    num_steps = 0
    start_time = time.time()

    # Step 1: Save original policy
    nn_policy_orig = nn_policy

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
            nn_policy_plus = _get_delta_policy(
                nn_policy, delta, params.delta_std)
            nn_policy_minus = _get_delta_policy(
                nn_policy, delta, -params.delta_std)

            # iii) Get rollouts
            sarss_plus = get_rollout(env, nn_policy_plus, False)
            sarss_minus = get_rollout(env, nn_policy_minus, False)
            num_steps += (len(sarss_plus) + len(sarss_minus))

            # iv) Estimate cumulative rewards
            r_plus = env.cum_reward(
                np.array([state for state, _, _, _ in sarss_plus]))
            r_minus = env.cum_reward(
                np.array([state for state, _, _, _ in sarss_minus]))

            # v) Save delta
            deltas.append((delta, r_plus, r_minus))

            # v) Update estimates of normalization parameters
            states = np.array([nn_policy.get_input(state)
                               for state, _, _, _ in sarss_plus + sarss_minus])
            mu_sum += np.sum(states)
            sigma_sq_sum += np.sum(np.square(states))
            n_states += len(states)

        # Step 3b: Sort deltas
        deltas.sort(key=lambda delta: -max(delta[1], delta[2]))
        deltas = deltas[:params.n_top_samples]

        # Step 3c: Compute the sum of the deltas weighted by their reward differences
        delta_sum = [torch.zeros(delta_cur.shape)
                     for delta_cur in deltas[0][0]]
        for j in range(params.n_top_samples):
            # i) Unpack values
            delta, r_plus, r_minus = deltas[j]

            # ii) Add delta to the sum
            for k in range(len(delta_sum)):
                delta_sum[k] += (r_plus - r_minus) * delta[k]

        # Step 3d: Compute standard deviation of rewards
        sigma_r = np.std([delta[1] for delta in deltas] +
                         [delta[2] for delta in deltas])

        # Step 3e: Compute step length
        delta_step = [(params.lr * params.delta_std / (params.n_top_samples * sigma_r + 1e-8))
                      * delta_sum_cur
                      for delta_sum_cur in delta_sum]

        # Step 3f: Update policy weights
        nn_policy = _get_delta_policy(nn_policy, delta_step, 1.0)

        # Step 3g: Update normalization parameters
        nn_policy.mu = mu_sum / n_states
        nn_policy.sigma_inv = 1.0 / np.sqrt((sigma_sq_sum / n_states))

        # Step 3h: Logging
        if i % params.log_interval == 0:
            exp_cum_reward, success_rate = test_policy(env, nn_policy, 100, use_cum_reward=True)
            current_time = time.time() - start_time
            print('\nSteps taken after iteration {}: {}'.format(i, num_steps))
            print('Reward after iteration {}: {}'.format(i, exp_cum_reward))
            print('Success rate after iteration {}: {}'.format(i, success_rate))
            print('Time after iteration {}: {} mins'.format(i, current_time/60))
            log_info.append([num_steps, current_time/60, exp_cum_reward, success_rate])

            # save best policy
            if success_rate > best_success_rate or (success_rate == best_success_rate
                                                    and exp_cum_reward >= best_reward):
                best_policy = nn_policy
                best_success_rate = success_rate
                best_reward = exp_cum_reward

            if success_rate > 80 and exp_cum_reward > 0:
                params.lr = max(params.lr/2, params.min_lr)

    nn_policy = best_policy

    # Step 4: Copy new weights and normalization parameters to original policy
    for param, param_orig in zip(nn_policy.parameters(), nn_policy_orig.parameters()):
        param_orig.data.copy_(param.data)
    nn_policy_orig.mu = nn_policy.mu
    nn_policy_orig.sigma_inv = nn_policy.sigma_inv

    return log_info


def _sample_delta(nn_policy):
    '''
    Construct random perturbations to neural network parameters.
    nn_policy: NNPolicy or NNPolicySimple
    Returns: [torch.tensor] (list of torch tensors that is the same shape as nn_policy.parameters())
    '''
    delta = []
    for param in nn_policy.parameters():
        delta.append(torch.normal(torch.zeros(param.shape, dtype=torch.float)))
    return delta


def _get_delta_policy(nn_policy, delta, sign):
    '''
    Construct the policy perturbed by the given delta

    Parameters:
        nn_policy: NNPolicy or NNPolicySimple
        delta: [torch.tensor] (list of torch tensors with same shape as nn_policy.parameters())
        sign: float

    Returns: NNPolicy or NNPolicySimple
    '''
    # Step 1: Construct the perturbed policy
    nn_policy_delta = None
    if (isinstance(nn_policy, NNPolicySimple)):
        nn_policy_delta = NNPolicySimple(nn_policy.params)
    elif (isinstance(nn_policy, NNPolicy)):
        nn_policy_delta = NNPolicy(nn_policy.params)
    else:
        raise Exception("Unrecognized neural network architecture")

    # Step 2: Set normalization of the perturbed policy
    nn_policy_delta.mu = nn_policy.mu
    nn_policy_delta.sigma_inv = nn_policy.sigma_inv

    # Step 3: Set the weights of the perturbed policy
    for param, param_delta, delta_cur in zip(nn_policy.parameters(), nn_policy_delta.parameters(),
                                             delta):
        param_delta.data.copy_(param.data + sign * delta_cur)

    return nn_policy_delta
