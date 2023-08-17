import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim import Adam
import gym
import time
import os.path as osp
from rlutil.logging import logger
import pickle
from scipy import spatial
import math
import os
# network


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPVFunction(nn.Module):

    def __init__(self, obs_dim, goal_dim, hidden_sizes, activation):
        super().__init__()
        self.v = mlp([obs_dim + goal_dim] +
                     list(hidden_sizes) + [1], activation)

    def forward(self, obs, goal):
        v = self.v(torch.cat([obs, goal], dim=-1))
        return torch.squeeze(v, -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, goal_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        goal_dim = goal_space.shape[0]

        # build value functions
        self.v = MLPVFunction(obs_dim, goal_dim, hidden_sizes, activation)

    def forward(self, obs, goal):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        return self.v.forward(obs, goal)


class MLPValueNetwork(nn.Module):

    def __init__(self, observation_space, goal_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        goal_dim = goal_space.shape[0]

        # build value functions
        self.v = mlp([obs_dim + goal_dim] +
                     list(hidden_sizes) + [1], activation)

    def forward(self, obs, goal):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        v = self.v(torch.cat([obs, goal], dim=-1))
        return torch.squeeze(v, -1)


# class ReplayBuffer:
#     """
#     A simple FIFO experience replay buffer for DDPG agents.
#     """

#     def __init__(self, obs_dim, act_dim, goal_dim, size):
#         self.obs_buf = np.zeros(combined_shape(
#             size, obs_dim), dtype=np.float32)
#         self.obs2_buf = np.zeros(combined_shape(
#             size, obs_dim), dtype=np.float32)
#         self.act_buf = np.zeros(combined_shape(
#             size, act_dim), dtype=np.float32)
#         self.goal_buf = np.zeros(combined_shape(
#             size, goal_dim), dtype=np.float32)
#         self.rew_buf = np.zeros(size, dtype=np.float32)
#         self.done_buf = np.zeros(size, dtype=np.float32)
#         self.ptr, self.size, self.max_size = 0, 0, size

#     def store(self, obs, act, goal, rew, next_obs, done):
#         self.obs_buf[self.ptr] = obs
#         self.obs2_buf[self.ptr] = next_obs
#         self.act_buf[self.ptr] = act
#         self.goal_buf[self.ptr] = goal
#         self.rew_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr+1) % self.max_size
#         self.size = min(self.size+1, self.max_size)

#     def sample_batch(self, batch_size=32):
#         idxs = np.random.randint(0, self.size, size=batch_size)
#         batch = dict(obs=self.obs_buf[idxs],
#                      obs2=self.obs2_buf[idxs],
#                      act=self.act_buf[idxs],
#                      goal=self.goal_buf[idxs],
#                      rew=self.rew_buf[idxs],
#                      done=self.done_buf[idxs])
#         return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, goal_dim, size):
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)
        self.goal_buf = np.zeros(combined_shape(
            size, goal_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.experienced = {}
        
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, goal, rew, next_obs, done, state_node,goal_node):
        # only keep the worst reward for each state-goal pair
        if tuple(state_node+goal_node) in self.experienced:
            experienced_ptr = self.experienced[tuple(state_node+goal_node)]
            if self.rew_buf[experienced_ptr] > rew:
                self.obs_buf[experienced_ptr] = obs
                self.obs2_buf[experienced_ptr] = next_obs
                self.act_buf[experienced_ptr] = act
                self.goal_buf[experienced_ptr] = goal
                self.rew_buf[experienced_ptr] = rew
                self.done_buf[experienced_ptr] = done
        else: 
            self.obs_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.act_buf[self.ptr] = act
            self.goal_buf[self.ptr] = goal
            self.rew_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)
            self.experienced[tuple(state_node+goal_node)] = self.ptr


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     goal=self.goal_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class value_policy:
    def __init__(self, env,
                 actor_critic=MLPValueNetwork,
                 hid=256,
                 l=3,
                 seed=0,
                 replay_size=int(1e6),
                 gamma=0.99,
                 polyak=0.995,
                 v_lr=1e-3,
                 batch_size=100,
                 update_every=10,
                 num_test_episodes=10,
                 logger_kwargs=dict(),
                 save_freq_steps=1000):

        self.env = env
        self.test_env = env
        self.actor_critic = actor_critic
        self.hid = hid
        self.l = l
        self.seed = seed
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.v_lr = v_lr
        self.batch_size = batch_size
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.logger_kwargs = logger_kwargs
        self.save_freq_steps = save_freq_steps

        torch.manual_seed(seed)
        np.random.seed(seed)

        # self.obs_dim = self.env.observation_space.shape
        self.obs_dim = self.env.wrapped_env.goal_space.shape[0]
        self.act_dim = self.env.wrapped_env.action_space.shape[0]
        self.goal_dim = self.env.wrapped_env.goal_space.shape[0]

        # Create actor-critic module and target networks
        ac_kwargs = dict(hidden_sizes=[self.hid]*self.l)
        # self.ac = actor_critic(self.env.observation_space, self.env.goal_space, ** ac_kwargs)
        # observation_space <- goal_space
        self.ac = actor_critic(
            self.env.wrapped_env.goal_space, self.env.wrapped_env.goal_space, ** ac_kwargs)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, goal_dim=self.goal_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(count_vars(module) for module in [self.ac.v])
        # logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

        # Set up optimizers for v-function
        self.v_optimizer = Adam(self.ac.v.parameters(), lr=self.v_lr)

    # Set up model saving

    # logger.setup_pytorch_saver(ac)
    def load_policy(self, filename):
        self.ac.load_state_dict(torch.load(filename))

    def update(self, data, node_covering_size):
        # First run one gradient descent step for Q.
        self.v_optimizer.zero_grad()
        loss_v, loss_info = self.compute_loss_v(data, node_covering_size)
        loss_v.backward()
        self.v_optimizer.step()
        return loss_v

        # Record things
        # logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # # Finally, update target networks by polyak averaging.
        # with torch.no_grad():
        #     for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
        #         # NB: We use an in-place operations "mul_", "add_" to update target
        #         # params, as opposed to "mul" and "add", which would make new tensors.
        #         p_targ.data.mul_(self.polyak)
        #         p_targ.data.add_((1 - self.polyak) * p.data)

    def execute(self, dataset, penalty, node_covering_size):

        # Reward-to-Go Policy Gradient
        rews = [t['rew'] for t in dataset]
        n = len(rews)
        rtgs = np.zeros_like(rews, dtype=np.float32)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + self.gamma * (rtgs[i + 1] if i + 1 < n else 0)
            # rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
            dataset[i]['rew'] = rtgs[i]
        
        # check tail for failed path
        if penalty:

            states = [t['obs'] for t in dataset]
            length = len(states)
            last_state = states[-1]
            penalty_dist_threshold = node_covering_size
            penalty_startidx = length-1
            for i in reversed(range(length)):
                if np.linalg.norm(states[i] - last_state) > penalty_dist_threshold:
                    break
                penalty_startidx = i

            # with open(os.getcwd()+"/gcsl_fetch/DebugLog.txt", "a") as f:
            #     f.write('from ' + str(penalty_startidx) +
            #             ' to ' + str(length)+":")
            #     f.write(str(dataset[penalty_startidx]['obs']))
            #     f.write(str(dataset[length-1]['obs'])+"\n")
            # f.close()

            for j in range(penalty_startidx, length):
                dataset[j]['rew'] =-1

        for i in range(n):
            t = dataset[i]
            # self.replay_buffer.store(
            #     t['obs'], t['act'], t['goal'], t['rew'], t['obs2'], t['done'])
            self.replay_buffer.store(
                t['obs'], t['act'], t['goal'], t['rew'], t['obs2'], t['done'], 
                self.get_node_by_state(t['obs'],node_covering_size), 
                self.get_node_by_state(t['goal'],node_covering_size))

        # Update handling
        total_loss = 0
        accumulate = 0
        for t in range(len(dataset)):
            if t % self.update_every == 0:
                batch = self.replay_buffer.sample_batch(self.batch_size)
                loss_v = self.update(batch, node_covering_size)
                accumulate += 1
                total_loss += loss_v
        avg_loss = total_loss/accumulate

        torch.save(
            self.ac.state_dict(),
            osp.join(logger.get_snapshot_dir(), 'value_policy.pkl')
        )
        with open(osp.join(logger.get_snapshot_dir(), 'replay_buffer.pkl'), 'wb') as f:
            pickle.dump(self.replay_buffer, f)
        return avg_loss
    # Set up function for computing v loss

    def get_node_by_state(self, state, node_covering_size):
        return (((state[0] + 0.5 * node_covering_size) // node_covering_size)*node_covering_size,
                ((state[1] + 0.5 * node_covering_size) // node_covering_size)*node_covering_size)

    def compute_loss_v(self, data, node_covering_size):
        o, a, g, r, o2, d = data['obs'], data['act'], data['goal'], data['rew'], data['obs2'], data['done']

        o_node = []
        g_node = []
        for i in range(len(o)):
            o_node.append(self.get_node_by_state(o[i], node_covering_size))
            g_node.append(self.get_node_by_state(g[i], node_covering_size))

        o_node = torch.as_tensor(o_node, dtype=torch.float32)
        g_node = torch.as_tensor(g_node, dtype=torch.float32)
        v = self.ac(o_node, g)

        # with open(os.getcwd()+"/gcsl_fetch/DebugLog.txt", "a") as f:
        #     f.write(str(o_node)+"\n")
        #     f.write(str(g)+"\n")
        #     f.write(str(r)+"\n")
        #     f.write(str(v)+"\n")
        # f.close()
        
        # MSE loss against Bellman backup
        loss_v = ((v - r)**2).mean()

        # Useful info for logging
        loss_info = dict(VVals=v.detach().numpy())

        return loss_v, loss_info

    def get_value(self, state_node, goal):
        '''
        state_node: int[2]
        goal: int[2]
        '''
        return self.ac.forward(state_node, goal)

    def end_epoch_handling(self, t):
        # End of epoch handling

        # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs):
        #     logger.save_state({'env': env}, None)

        # Test the performance of the deterministic version of the agent.
        # self.test_agent()

        # Log info about epoch
        # logger.log_tabular('Epoch', epoch)
        # logger.log_tabular('EpRet', with_min_and_max=True)
        # logger.log_tabular('TestEpRet', with_min_and_max=True)
        # logger.log_tabular('EpLen', average_only=True)
        # logger.log_tabular('TestEpLen', average_only=True)
        # logger.log_tabular('TotalEnvInteracts', t)
        # logger.log_tabular('QVals', with_min_and_max=True)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('LossQ', average_only=True)
        # logger.log_tabular('Time', time.time()-start_time)
        # logger.dump_tabular()
        return

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
