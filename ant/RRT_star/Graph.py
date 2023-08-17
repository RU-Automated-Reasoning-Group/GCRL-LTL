import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc
from collections import deque
from RRT_star.utils import *
import warnings
import random


class Graph:
    ''' Define graph 
    currently only for AntPush Env
    '''

    def __init__(self, startpos, endpos, algo='Dijkstra'):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.vex2idx = {startpos: 0}
        self.neighbors = {0: []}

        self.reward_list = {}

        self.distance = []
        self.predecessor = []

        self.success = False
        self.algo = algo

    def set_algo(self, algo):
        self.algo = algo

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, childrenidx, parentidx, edgecost):
        if (childrenidx, parentidx, edgecost) not in self.edges:
            self.edges.append((childrenidx, parentidx, edgecost))
            self.neighbors[childrenidx].append((parentidx, edgecost))
            self.neighbors[parentidx].append((childrenidx, edgecost))

    # updating shortest path algo1
    def BellmanFord(self, source_idx, goal, value_policy):
        # bellman-ford algo
        negative_weight_cycle = False
        self.distance.clear()
        self.predecessor.clear()
        V = len(self.vertices)

        # initialize graph
        for i in range(V):
            self.distance.append(float('inf'))
            self.predecessor.append(-1)
        self.distance[source_idx] = 0

        values_cache = {}

        # relax edges repeatedly |V|-1 times
        for i in range(V-1):
            for edge in self.edges:
                # edge (u,v) from u ot v
                u = edge[1]
                v = edge[0]

                if v in values_cache.keys():
                    value = values_cache[v]
                else:
                    value = value_policy.get_value(self.vertices[v], goal)
                    values_cache[v] = value

                if value > 0:
                    value = 0
                weight = edge[2] - value
                # weight = 0 - value_policy.get_value(self.vertices[v], goal)
                # if weight < 0:
                #     weight = 0

                if self.distance[u] + weight < self.distance[v]:
                    self.distance[v] = self.distance[u] + weight
                    self.predecessor[v] = u

        # check for negative-weight cycles
        for edge in self.edges:
            u = edge[1]
            v = edge[0]
            if v in values_cache.keys():
                value = values_cache[v]
            else:
                value = value_policy.get_value(self.vertices[v], goal)
                values_cache[v] = value

            if value > 0:
                value = 0
            # weight = edge[2] - value
            weight = - value

            if self.distance[u] != float('inf') and self.distance[u] + weight < self.distance[v]:
                negative_weight_cycle = True
                break

        return negative_weight_cycle

    # # updating shortest path algo2 with reward -[1/0]
    # def Dijkstra(self, source_idx, goal, value_policy):
    #     '''
    #     Dijkstra algorithm for finding shortest path from start position to end.
    #     '''

    #     # build dijkstra
    #     nodes = list(self.neighbors.keys())
    #     self.distance.clear()
    #     self.predecessor.clear()
    #     values_cache = {}
    #     # initialize graph
    #     for node in self.vertices:
    #         self.distance.append(float('inf'))
    #         self.predecessor.append(-1)
    #     self.distance[source_idx] = 0

    #     while nodes:
    #         curNode = min(nodes, key=lambda node: self.distance[node])
    #         nodes.remove(curNode)
    #         if self.distance[curNode] == float('inf'):
    #             break

    #         for neighbor, cost in self.neighbors[curNode]:
    #             if neighbor in values_cache.keys():
    #                 value = values_cache[neighbor]
    #             else:
    #                 value = value_policy.get_value(
    #                     self.vertices[neighbor], goal)
    #                 values_cache[neighbor] = value

    #             if value > 0:
    #                 value = 0
    #             # weight = cost - value
    #             weight = - value
    #             newCost = self.distance[curNode] + weight
    #             if newCost < self.distance[neighbor]:
    #                 self.distance[neighbor] = newCost
    #                 self.predecessor[neighbor] = curNode

    #     return

    # updating shortest path algo2 with reward[0/1]
    def Dijkstra(self, source_idx, goal, value_policy):
        '''
        Dijkstra algorithm for finding shortest path from start position to end.
        '''

        # build dijkstra
        nodes = list(self.neighbors.keys())
        self.distance.clear()
        self.predecessor.clear()
        # initialize graph
        for node in self.vertices:
            self.distance.append(float('inf'))
            self.predecessor.append(-1)
        self.distance[source_idx] = 0

        while nodes:
            curNode = min(nodes, key=lambda node: self.distance[node])
            nodes.remove(curNode)
            if self.distance[curNode] == float('inf'):
                break

            for neighbor, cost in self.neighbors[curNode]:
                value = value_policy.get_value(self.vertices[neighbor], goal)
                # value = value_policy.get_value(self.vertices[curNode],self.vertices[neighbor])

                weight = 1 - value
                if weight < 0:
                    weight = 0
                newCost = self.distance[curNode] + weight
                if newCost < self.distance[neighbor]:
                    self.distance[neighbor] = newCost
                    self.predecessor[neighbor] = curNode

        return
    
    def get_closest_node_by_distance(self,root_nodeidx,goal):
        root_node = self.vertices[root_nodeidx]
        closest_nodeidx = root_nodeidx
        # opt[2] get closest_nodeidx by norm-2 xy distance
        dist = distance(root_node, goal)
        for state_node in self.vertices:
            if distance(state_node, goal) < dist:
                dist = distance(state_node, goal)
                closest_nodeidx = self.vex2idx[state_node]
        return closest_nodeidx

    def get_closest_node_by_value(self,root_nodeidx,goal,value_policy):

        root_node = self.vertices[root_nodeidx]
        closest_nodeidx = root_nodeidx
        # opt[1] get closest_nodeidx by v
        dist = - value_policy.get_value(root_node, goal)
        for state_node in self.vertices:
            if - value_policy.get_value(state_node, goal) < dist:
                dist = - value_policy.get_value(state_node, goal)
                closest_nodeidx = self.vex2idx[state_node]

        return closest_nodeidx
    
    def find_path_to_closest_node(self, root_nodeidx, goal, value_policy, closest_node_mode="by_value"):
        # 1. find the node that is the closest to the goal
        if closest_node_mode=="by_value":
            closest_nodeidx = self.get_closest_node_by_value(root_nodeidx=root_nodeidx, goal=goal, value_policy=value_policy)
        else:
            closest_nodeidx=self.get_closest_node_by_distance(root_nodeidx=root_nodeidx,goal=goal)

        # 2. update shortest path to ant other node from given start node
        if self.algo == 'Dijkstra':
            # Dijkstra's algo
            self.Dijkstra(root_nodeidx, goal, value_policy)
        else:
            # Bellman_Ford algo:
            negative_weight_cycle = self.BellmanFord(
                root_nodeidx, goal, value_policy)
            if negative_weight_cycle == True:
                warnings.warn(
                    'Warning message: negative_weight_cycle in graph')

        # 3.get path from init node to the closest node
        res = []
        cur_nodeidx = closest_nodeidx
        while cur_nodeidx != -1:
            res.append(cur_nodeidx)
            cur_nodeidx = self.predecessor[cur_nodeidx]

        # with open("./code/gcsl_ant/DebugLog.txt", "a") as f:
        #     f.write(str(goal)+"\n")
        #     f.write(str(self.vertices[closest_nodeidx])+"\n")
        #     f.write(str(len(res))+"\n")
        # f.close()
        return res

    def get_random_start_pos(self):
        pos = np.random.randint(len(self.vertices))
        while self.neighbors[pos] == []:
            pos = np.random.randint(len(self.vertices))
        return self.vertices[pos]
