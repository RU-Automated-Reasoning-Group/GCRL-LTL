from types import SimpleNamespace
from collections import OrderedDict

from algo.ltl import gltl2ba, PathGraph


INF = 999
NO_PARENT = -1
FOREVER = 10000
OMEGA = 5


def get_ltl_args(formula):
    
    args = SimpleNamespace()
    args.formula = formula
    args.file = None

    args.s = False
    args.d = False
    args.l = False
    args.p = False
    args.o = False
    args.c = False
    args.a = False

    args.output_graph = None
    args.output_dot = None

    return args


def reformat_ltl(formula):
    return formula.replace('F', '<>').replace('G', '[]')


class SCCAlgorithm:
    def __init__(self, graph):
        self.storage = OrderedDict()
        self.accepting_nodes = graph.accepting_nodes
        for node in graph.iternodes():
            self.storage[node] = OrderedDict({'next': [], 'edges': []})
            for edge in graph.iteroutedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ')
                self.storage[src]['next'].append(dst)
                self.storage[src]['edges'].append(f)
        self.graph_keys = self.storage.keys()
        self.visited = []
        self.stack = []
        self.time = 0
        self.d = dict(zip(self.graph_keys, [None]*len(self.graph_keys)))  # discovery time
        self.lowlink = dict(zip(self.graph_keys, [None]*len(self.graph_keys)))
        self.SCCS = []

    def SCC_search(self, v: str):
        self.visited.append(v)
        self.d[v] = self.time
        self.time += 1
        self.lowlink[v] = self.d[v]
        self.stack.append(v)
        for next_v in self.storage[v]['next']:
            if next_v not in self.visited:
                self.SCC_search(next_v)
                self.lowlink[v] = min(self.lowlink[v], self.lowlink[next_v])
            elif self.d[next_v] < self.d[v] and next_v in self.stack:
                self.lowlink[v] = min(self.d[next_v], self.lowlink[v])
        if self.lowlink[v] == self.d[v]:
            scc = []
            accepting = False
            while True:
                x = self.stack.pop()
                if x in self.accepting_nodes:
                    accepting = True
                scc.append(x)
                if x == v:
                    self.SCCS.append([scc, accepting])
                    break
        
    def search(self, v=None):
        if v is None:
            for name in self.graph_keys:
                if 'init' in name:
                    v = name
        self.SCC_search(v)
        sccs = [scc[0] for scc in self.SCCS if scc[1]]
        
        return sccs


class PathFindingAlgorithm:
    def __init__(self, graph, value_map, path_type='direct'):
        self.graph = graph
        self.value_map = value_map
        assert path_type in ('direct', 'loop')
        self.path_type = path_type
        self.nodes = []
        self.storage = OrderedDict()
        for node in self.graph.iternodes():
            self.storage[node] = OrderedDict({'next': [], 'edges': []})
            self.nodes.append(node)
            for edge in self.graph.iteroutedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ')
                self.storage[src]['next'].append(dst)
                self.storage[src]['edges'].append(f)
        self.build_matrix()
    
    def compute_cost(self, src, dst):
        src, dst = src.split('|')[1].upper(), dst.split('|')[1].upper()
        if src == 'EMPTY':
            cost = self.value_map[str(dst)]
        else:
            cost = self.value_map[str(src + dst)]
        
        return cost.item()

    def build_matrix(self):
        self.matrix = [[INF] * len(self.nodes) for _ in range(len(self.nodes))]
        for node in self.nodes:
            row = self.matrix[self.node_to_index(node)]
            if self.path_type == 'direct':
                row[self.node_to_index(node)] = 0
            elif self.path_type == 'loop':
                row[self.node_to_index(node)] = INF
            for edge in self.graph.iteroutedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ')
                print('[DEBUG]', src, dst)
                # TODO: estimate the cost
                zero_cost = dst.split('|')[1] == '1' or '!' in dst.split('|')[1]
                row[self.node_to_index(dst)] = 0 if zero_cost else 1 #self.compute_cost(src, dst)

    def update_matrix(self):
        raise NotImplementedError

    def node_to_index(self, node):
        return self.nodes.index(node)

    def index_to_node(self, index):
        return self.nodes[index]
    
    def dijkstra(self, start_vertex):
        
        n_vertices = len(self.matrix[0])
     
        # added[i] will true if vertex i is
        # included in shortest path tree
        # or shortest distance from start_vertex to
        # i is finalized
        added = [False] * n_vertices
    
        # Initialize all added[] as false
        for vertex_index in range(n_vertices):
            added[vertex_index] = False
            
        # Distance of source vertex from itself is not always 0
        shortest_distances = self.matrix[start_vertex]

        # Parent array to store shortest path tree
        parents = [-1] * n_vertices
    
        # The starting vertex does not have a parent
        parents[start_vertex] = NO_PARENT
    
        # Find shortest path for all vertices
        for i in range(1, n_vertices):
            # Pick the minimum distance vertex from the set of vertices not yet processed
            # nearest_vertex is always equal to start_vertex in first iteration.
            nearest_vertex = -1
            shortest_distance = INF
            for vertex_index in range(n_vertices):
                if not added[vertex_index] and shortest_distances[vertex_index] < shortest_distance:
                    nearest_vertex = vertex_index
                    shortest_distance = shortest_distances[vertex_index]
    
            # Mark the picked vertex as processed
            added[nearest_vertex] = True
    
            # Update dist value of the adjacent vertices of the picked vertex
            for vertex_index in range(n_vertices):
                edge_distance = self.matrix[nearest_vertex][vertex_index]
                
                if shortest_distance + edge_distance < shortest_distances[vertex_index]:
                    parents[vertex_index] = nearest_vertex
                    shortest_distances[vertex_index] = shortest_distance + edge_distance
    
        self.start_vertex = start_vertex
        self.distances = shortest_distances
        self.parents = parents

    def get_path(self, current_vertex, parents, path):
        if current_vertex == NO_PARENT:
            return path
        self.get_path(parents[current_vertex], parents, path)
        path.append(self.index_to_node(current_vertex))
        
        return path

    def get_accepting_path(self, accepting_nodes):

        assert self.path_type == 'direct'
        
        start_vertex = None
        for node in self.nodes:
            if 'init' in node:
                start_vertex = self.node_to_index(node)
                break
        assert not start_vertex is None

        self.dijkstra(start_vertex)

        acc_paths = {}
        n_vertices = len(self.distances)
        for vertex_index in range(n_vertices):
            node = self.index_to_node(vertex_index)
            if node in accepting_nodes:
                path = self.get_path(vertex_index, self.parents, path=[])
                acc_paths[node] = {'path': path, 'cost': self.distances[vertex_index], 'plan': self.build_plan(path)}
        
        return acc_paths
    
    def get_loop_path(self, accepting_nodes):

        assert self.path_type == 'loop'

        loop_paths = {}
        for node in accepting_nodes:
            node_index = self.node_to_index(node)
            self.dijkstra(start_vertex=node_index)
            path = self.get_path(node_index, self.parents, path=[])
            loop_paths[node] = {'path': path, 'cost': self.distances[node_index], 'plan': self.build_plan(path)}
        
        return loop_paths

    def build_plan(self, path):

        if self.path_type == 'direct':
            path = [self.index_to_node(self.start_vertex)] + path
        
        elif self.path_type == 'loop':
            path = [path[-1]] + path

        GOALS = []
        AVOIDS = []

        for p_idx, node in enumerate(path):
            
            if p_idx == 0:
                continue

            # TODO: more robust implementation
            state = node.split('|')[-1].capitalize()
            prev_node = path[p_idx - 1]
            edge = self.graph.get_edge(prev_node, node)
            f = edge.get_name()
            if '{}' in f:
                avoid = []
            else:
                avoid = edge.get_name().replace('&&', '').replace('!', '').split()
                avoid = [ap.capitalize() for ap in avoid]
            
            if state not in ('empty', '1') and '!' not in state:
                GOALS.append(state)
                AVOIDS.append(avoid)

        return GOALS, AVOIDS


def path_finding(formula, value_map, debug=False):

    formula = reformat_ltl(formula)
    ltl_args = get_ltl_args(formula=formula)
    buchi_graph = gltl2ba(ltl_args)
    if debug:
        buchi_graph.save('buchi.png')

    path_graph = PathGraph()
    path_graph.build(buchi_graph)
    if debug:
        path_graph.save('path_finding.png')
    
    scc_algo = SCCAlgorithm(graph=path_graph)
    sccs = scc_algo.search()

    plans = []
    for scc in sccs:
        if debug:
            print('-' * 80)
            print('[searching scc]', scc)
        accepting_nodes=[node for node in scc if 'accept' in node]
        
        path_algo = PathFindingAlgorithm(graph=path_graph, value_map=value_map, path_type='direct')
        acc_paths = path_algo.get_accepting_path(accepting_nodes=accepting_nodes)
        if debug:
            print('[acc_paths]', acc_paths)

        if len(scc) <= 1:
            GOALS = acc_paths[accepting_nodes[0]]['plan'][0]
            AVOIDS = acc_paths[accepting_nodes[0]]['plan'][1]
            cost = acc_paths[accepting_nodes[0]]['cost']
        else:
            scc_graph = path_graph.build_sub_graph(sub_graph_nodes=scc)
            loop_algo = PathFindingAlgorithm(graph=scc_graph, value_map=value_map, path_type='loop')
            loop_paths = loop_algo.get_loop_path(accepting_nodes=accepting_nodes)
            if debug:
                scc_graph.save('scc_path_finding.png')
                print('[loop_paths]', loop_paths)
            
            min_cost_acc_node = None
            min_cost = INF

            for node in accepting_nodes:
                p_cost, q_cost = acc_paths[node]['cost'], loop_paths[node]['cost']
                cost = p_cost + OMEGA * q_cost
                if cost < min_cost:
                    min_cost = cost
                    min_cost_acc_node = node
            
            GOALS = acc_paths[min_cost_acc_node]['plan'][0] + loop_paths[min_cost_acc_node]['plan'][0] * FOREVER
            AVOIDS = acc_paths[min_cost_acc_node]['plan'][1] + loop_paths[min_cost_acc_node]['plan'][1] * FOREVER
        plans.append((cost, GOALS, AVOIDS))
    
    plans.sort()
    GOALS, AVOIDS = plans[0][1], plans[0][2]

    return GOALS, AVOIDS


if __name__ == '__main__':

    f1 = '(!p U d) && (!e U (q && (!n U a)))'
    f2 = '<>b && a U b'  # cannot be solved
    f3 = '!j U (w && (!y U r))'
    f4 = 'Fa'
    f5 = 'GFa'
    f6 = 'GFa && GFb'
    f7 = '[]<>a && []<>b'
    f8 = 'GF(r && XF y) && G(!w)'
    f9 = '[](<>(o && X (<> (c && X<> d))))'
    f10 = '(! w) U ( r && ((! y) U j)) U (! y)'
    f11 = '(! w) U ( r && ((! y) U j))'
    f12 = '<>((b || q) && <>((e || p) && <>m))'
    f13 = '<>((c || n) && <>(r && <>d)) && <>(q && <>((r || t) && <>m))'
    f14 = '!y U (j && (!w U r))'
    f15 = 'F((a_1 || a_2) && F(b && F((c_1 || c_2) && F(d && F((e_1 || e_2) && F(z && F((k_1 || k_2) && F(h))))))))'
    f16 = 'GF(a_1 && XF(a_2 && XF(a_3 && XF(a_4))))'
    f17 = 'GF(room_1_0 && XF(room_3_0 && XF(room_3_2 && XF(room_1_2))))'
    f18 = 'Froom_0_2 && XGF(room_2_2 && XF(room_3_2 && XF(room_3_3 && XF(room_2_3))))'
    f19 = 'GF( room_1_0 && XF( room_3_0 && XF(room_3_2 && XF(room_1_2)))) || (F room_0_2 && XGF( room_2_2 && XF( room_3_2 && XF( room_3_3 && XF( room_2_3)))))'

    formula = f1
    print('[INPUT FORMULA]', formula)
    
    goals, avoid_zones = path_finding(formula, debug=True)
    print('-' * 80)
    print('[GOALS]', goals)
    print('[AVOID]', avoid_zones)
