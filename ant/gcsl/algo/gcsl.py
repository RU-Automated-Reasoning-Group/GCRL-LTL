import numpy as np
from torch import threshold
from rlutil.logging import logger

import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu

import time
import tqdm
import os.path as osp
import copy
import pickle
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_enabled = True
except:
    print('Tensorboard not installed!')
    tensorboard_enabled = False


class GCSL:
    """Goal-conditioned Supervised Learning (GCSL).

    Parameters:
        env: A gcsl.envs.goal_env.GoalEnv
        policy: The policy to be trained (likely from gcsl.algo.networks)
        replay_buffer: The replay buffer where data will be stored
        validation_buffer: If provided, then 20% of sampled trajectories will
            be stored in this buffer, and used to compute a validation loss
        max_timesteps: int, The number of timesteps to run GCSL for.
        max_path_length: int, The length of each trajectory in timesteps

        # Exploration strategy

        explore_timesteps: int, The number of timesteps to explore randomly
        expl_noise: float, The noise to use for standard exploration (eps-greedy)

        # Evaluation / Logging Parameters

        goal_threshold: float, The distance at which a trajectory is considered
            a success. Only used for logging, and not the algorithm.
        eval_freq: int, The policy will be evaluated every k timesteps
        eval_episodes: int, The number of episodes to collect for evaluation.
        save_every_iteration: bool, If True, policy and buffer will be saved
            for every iteration. Use only if you have a lot of space.
        log_tensorboard: bool, If True, log Tensorboard results as well

        # Policy Optimization Parameters

        start_policy_timesteps: int, The number of timesteps after which
            GCSL will begin updating the policy
        batch_size: int, Batch size for GCSL updates
        n_accumulations: int, If desired batch size doesn't fit, use
            this many passes. Effective batch_size is n_acc * batch_size
        policy_updates_per_step: float, Perform this many gradient updates for
            every environment step. Can be fractional.
        train_policy_freq: int, How frequently to actually do the gradient updates.
            Number of gradient updates is dictated by `policy_updates_per_step`
            but when these updates are done is controlled by train_policy_freq
        lr: float, Learning rate for Adam.
        demonstration_kwargs: Arguments specifying pretraining with demos.
            See GCSL.pretrain_demos for exact details of parameters
    """

    def __init__(self,
                 env,
                 policy,
                 replay_buffer,
                 validation_buffer=None,
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
                 demonstrations_kwargs=dict(),
                 lr=5e-4,
                 ):
        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.validation_buffer = validation_buffer

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
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.log_tensorboard = log_tensorboard and tensorboard_enabled
        self.summary_writer = None

    def loss_fn(self, observations, goals, actions, horizons, weights):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if self.is_discrete_action else torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype).cuda()
        goals_torch = torch.tensor(goals, dtype=obs_dtype).cuda()
        actions_torch = torch.tensor(actions, dtype=action_dtype).cuda()
        horizons_torch = torch.tensor(horizons, dtype=obs_dtype).cuda()
        weights_torch = torch.tensor(weights, dtype=torch.float32).cuda()
        # observations_torch

        conditional_nll = self.policy.nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
        nll = conditional_nll

        return torch.mean(nll * weights_torch)

    def sample_trajectory(self, greedy=False, noise=0, render=False, given_goal="none"):
        if given_goal == "none":
            goal_state = self.env.sample_goal()
            goal = self.env.extract_goal(goal_state)
        elif given_goal == "antumaze":
            goal_state = np.zeros(30)
            goal_state[:2] = np.array([0, 16])
            goal = self.env.extract_goal(goal_state)
        elif given_goal == "antfall":
            goal_state = np.zeros(33)
            goal_state[:2] = np.array([0, 27])
            goal = self.env.extract_goal(goal_state)

        # keep tracking of trajectory
        states = []
        actions = []

        state = self.env.reset()
        for t in range(self.max_path_length):
            if render:
                self.env.render()

            states.append(state)

            # get observation
            observation = self.env.observation(state)
            horizon = np.arange(self.max_path_length) >= (self.max_path_length - 1 - t)  # Temperature encoding of horizon
            action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
            # print("action:" + str(action))
            if not self.is_discrete_action:
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            actions.append(action)
            # execute action on current state
            state, _, done, _ = self.env.step(action)

            if done == False:
                done = True if np.linalg.norm((states[-1][:2] - goal), axis=-1) < self.goal_threshold else False

            if done == True:
                # states.append(state)
                break

        trajectory_length = len(states)

        # pad the states and actions to be the same size (max_path_length * state.shape/action.shape)
        if trajectory_length < self.max_path_length:
            pad_len = self.max_path_length-trajectory_length
            pad_state = states[-1]
            pad_action = actions[-1]
            for i in range(pad_len):
                states.append(pad_state)
                actions.append(pad_action)

        return np.stack(states), np.array(actions), goal_state, trajectory_length

    def take_policy_step(self, buffer=None):
        if buffer is None:
            buffer = self.replay_buffer

        avg_loss = 0
        self.policy_optimizer.zero_grad()

        for _ in range(self.n_accumulations):
            observations, actions, goals, _, horizons, weights = buffer.sample_batch(self.batch_size)
            loss = self.loss_fn(observations, goals, actions, horizons, weights)
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
            observations, actions, goals, lengths, horizons, weights = buffer.sample_batch(self.batch_size)
            loss = self.loss_fn(observations, goals, actions, horizons, weights)
            avg_loss += ptu.to_numpy(loss)

        return avg_loss / self.n_accumulations

    def pretrain_demos(self, demo_replay_buffer=None, demo_validation_replay_buffer=None, demo_train_steps=0):
        if demo_replay_buffer is None:
            return

        self.policy.train()
        with tqdm.trange(demo_train_steps) as looper:
            for _ in looper:
                loss = self.take_policy_step(buffer=demo_replay_buffer)
                validation_loss = self.validation_loss(buffer=demo_validation_replay_buffer)

                if running_loss is None:
                    running_loss = loss
                else:
                    running_loss = 0.99 * running_loss + 0.01 * loss
                if running_validation_loss is None:
                    running_validation_loss = validation_loss
                else:
                    running_validation_loss = 0.99 * running_validation_loss + 0.01 * validation_loss

                looper.set_description('Loss: %.03f Validation Loss: %.03f' % (running_loss, running_validation_loss))

    def train(self):
        start_time = time.time()
        last_time = start_time
        sample_time = 0

        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0
        timesteps_since_reset = 0

        iteration = 0
        running_loss = None
        running_validation_loss = None

        if logger.get_snapshot_dir() and self.log_tensorboard:
            self.summary_writer = SummaryWriter(osp.join(logger.get_snapshot_dir(), 'tensorboard'))

        # Evaluation Code
        self.policy.eval()
        self.evaluate_policy(self.eval_episodes, total_timesteps=0, greedy=True, prefix='Eval')
        logger.record_tabular('policy loss', 0)
        logger.record_tabular('timesteps', total_timesteps)
        logger.record_tabular('epoch time (s)', time.time() - last_time)
        logger.record_tabular('total time (s)', time.time() - start_time)
        logger.record_tabular('gen sample trajectory time (s)', sample_time)
        last_time = time.time()
        logger.dump_tabular()
        # End Evaluation Code

        ###
        sample_goals_x = []
        sample_goals_y = []
        ###

        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:

                # Interact in environmenta according to exploration strategy.
                start_sample_time = time.time()
                if total_timesteps < self.explore_timesteps:
                    states, actions, goal_state, trajectory_len = self.sample_trajectory(noise=1)
                else:
                    states, actions, goal_state, trajectory_len = self.sample_trajectory(greedy=True, noise=self.expl_noise)
                end_sample_time = time.time()
                sample_time += end_sample_time - start_sample_time
                ###
                sample_goals_x.append(goal_state[0])
                sample_goals_y.append(goal_state[1])
                ###

                # With some probability, put this new trajectory into the validation buffer
                assert len(states) == len(actions)
                if self.validation_buffer is not None and np.random.rand() < 0.2:
                    self.validation_buffer.add_trajectory(states, actions, goal_state, length_of_traj=trajectory_len)
                else:
                    self.replay_buffer.add_trajectory(states, actions, goal_state, length_of_traj=trajectory_len)

                total_timesteps += self.max_path_length
                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length

                ranger.update(self.max_path_length)

                # Take training steps
                if timesteps_since_train >= self.train_policy_freq and total_timesteps > self.start_policy_timesteps:
                    timesteps_since_train %= self.train_policy_freq
                    # train
                    self.policy.train()
                    for _ in range(int(self.policy_updates_per_step * self.train_policy_freq)):
                        loss = self.take_policy_step()
                        validation_loss = self.validation_loss()
                        if running_loss is None:
                            running_loss = loss
                        else:
                            running_loss = 0.9 * running_loss + 0.1 * loss

                        if running_validation_loss is None:
                            running_validation_loss = validation_loss
                        else:
                            running_validation_loss = 0.9 * running_validation_loss + 0.1 * validation_loss

                    self.policy.eval()
                    ranger.set_description('Loss: %s Validation Loss: %s' % (running_loss, running_validation_loss))

                    if self.summary_writer:
                        self.summary_writer.add_scalar('Losses/Train', running_loss, total_timesteps)
                        self.summary_writer.add_scalar('Losses/Validation', running_validation_loss, total_timesteps)

                # Evaluate, log, and save to disk
                if timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    iteration += 1
                    # Evaluation Code
                    self.policy.eval()
                    self.evaluate_policy(self.eval_episodes, total_timesteps=total_timesteps, greedy=True, prefix='Eval')
                    logger.record_tabular('policy loss', running_loss or 0)  # Handling None case
                    logger.record_tabular('timesteps', total_timesteps)
                    logger.record_tabular('epoch time (s)', time.time() - last_time)
                    logger.record_tabular('total time (s)', time.time() - start_time)
                    logger.record_tabular('gen sample trajectory time (s)', sample_time)
                    last_time = time.time()
                    logger.dump_tabular()

                    # Logging Code
                    if logger.get_snapshot_dir():
                        modifier = str(iteration) if self.save_every_iteration else ''
                        torch.save(
                            self.policy.state_dict(),
                            osp.join(logger.get_snapshot_dir(), 'policy%s.pkl' % modifier)
                        )
                        if hasattr(self.replay_buffer, 'state_dict'):
                            with open(osp.join(logger.get_snapshot_dir(), 'buffer%s.pkl' % modifier), 'wb') as f:
                                pickle.dump(self.replay_buffer.state_dict(), f)

                        full_dict = dict(env=self.env, policy=self.policy)
                        with open(osp.join(logger.get_snapshot_dir(), 'params%s.pkl' % modifier), 'wb') as f:
                            pickle.dump(full_dict, f)

                    ranger.reset()

        self.env.sample_goal_scatter_fig(sample_goals_x, sample_goals_y)

    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval', total_timesteps=0):
        env = self.env

        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state, trajectory_len = self.sample_trajectory(noise=0, greedy=greedy)
            assert len(states) == len(actions)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(states[trajectory_len-1], goal_state)

            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        logger.record_tabular('%s num episodes' % prefix, eval_episodes)
        logger.record_tabular('%s avg final dist' % prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio' % prefix, np.mean(success_vec))
        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist' % prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio' % prefix,  np.mean(success_vec), total_timesteps)
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s' % (prefix, key), value)

        return all_states, all_goal_states

    def visualize_evaluate_policy(self, eval_episodes, greedy=True, prefix='Eval', total_timesteps=0, env_type="none"):
        env = self.env

        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state, trajectory_len = self.sample_trajectory(noise=0, greedy=greedy, given_goal=env_type)
            assert len(states) == len(actions)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(states[trajectory_len-1], goal_state)

            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        print('%s/success ratio' % prefix,  np.mean(success_vec))
        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        logger.record_tabular('%s num episodes' % prefix, eval_episodes)
        logger.record_tabular('%s avg final dist' % prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio' % prefix, np.mean(success_vec))
        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist' % prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio' % prefix,  np.mean(success_vec), total_timesteps)
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s' % (prefix, key), value)

        return all_states, all_goal_states
