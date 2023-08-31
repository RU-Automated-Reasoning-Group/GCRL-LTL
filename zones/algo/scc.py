from types import SimpleNamespace
from collections import OrderedDict

import spot

#from algo.ltl import gltl2ba
from ltl import gltl2ba

FOREVER = 10
OMEGA = 5
INF = 9999


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
    f = spot.formula(formula)
    f = str(f).replace('&', '&&').replace('"', '').replace('|', '||').lower()
    f = f.replace('u', 'U').replace('f', '<>').replace('g', '[]').replace('x', 'X')

    return f


def ltl_to_zones(ltl, translation_function=None):

    if translation_function is None:
        def translate(word):
            return word.capitalize()
    else:
        translate = translation_function

    GOALS, AVOID_ZONES = [], []
    for f in ltl:
        avoid_zones = []
        f = f.replace('(', '').replace(')', '').split('&&')
        f = [_f.strip() for _f in f]
        for _f in f:
            if '!' not in _f:
                GOALS.append(translate(_f))
            else:
                avoid_zones.append(translate(_f.replace('!', '')))
        AVOID_ZONES.append(avoid_zones)

    while '1' in GOALS:
        all_index = GOALS.index('1')
        GOALS.pop(all_index)
        AVOID_ZONES.pop(all_index)

    return GOALS, AVOID_ZONES


class SCCAlgorithmGraph:
    def __init__(self, graph):
        self.storage = OrderedDict()
        self.accepting_nodes = graph.accepting_nodes
    
        for node in graph.iternodes():
            self.storage[node] = OrderedDict({'next': [], 'edges': []})
            for edge in graph.iteroutedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ')
                self.storage[src]['next'].append(dst)
                self.storage[src]['edges'].append(f)


class DijkstraAlgorithm:
    def __init__(self, graph, scc):
        self.graph = graph
        self.scc = scc
        # graph nodes/edge information
        self.storage = OrderedDict()
        self.nodes = []
        for node in self.graph.iternodes():
            self.storage[node] = OrderedDict({'next': [], 'edges': []})
            self.nodes.append(node)
            for edge in self.graph.iteroutedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ')
                self.storage[src]['next'].append(dst)
                self.storage[src]['edges'].append(f)
        self.graph_keys = self.storage.keys()
    
    def node_to_index(self, node):
        return self.nodes.index(node)
        
    def reset(self, v):
        # graph geological information
        self.matrix = [[INF] * len(self.nodes) for _ in range(len(self.nodes))]
        for node in self.nodes:
            node_index = self.node_to_index(node)
            row = self.matrix[node_index]
            row[node_index] = 0
            for edge in self.graph.iteroutedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ')
                row[self.node_to_index(dst)] = 0 if f == '(1)' else 1
        for e in self.matrix:
            print(e)
        exit()
                
        #self.storage[src]['weights'].append(0 if f == '(1)' else 0)  # NOTE: assume equal 1 weights for now
        self.graph_keys = self.storage.keys()
        self.visited = []
        self.distance = OrderedDict(zip(self.graph_keys, [INF]*len(self.graph_keys)))
        self.visited.append(v)
        self.distance[v] = 0

    def find_nearest_node(self):
        pass

    def search(self, v=None):
        
        if v is None:
            for name in self.graph_keys:
                if 'init' in name:
                    v = name
        
        self.reset(v)
    
        for node in self.graph.iternodes():
            print('[NODE]', node)

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

    def DFS_search(self, v, goal_scc=None):
        for scc in self.SCCS:
            if scc[1]:
                goal_scc = scc[0]
        stack = [(v, [v])]
        visited = set()
        while stack:
            (vertex, path) = stack.pop()
            if vertex not in visited:
                if vertex in goal_scc:
                    return {'nodes': path, 'scc': goal_scc}
                visited.add(vertex)
                for neighbor in self.storage[vertex]['next']:
                    stack.append((neighbor, path + [neighbor]))
        
    def search(self, v=None):
        if v is None:
            for name in self.graph_keys:
                if 'init' in name:
                    v = name
        self.SCC_search(v)
        info = self.DFS_search(v)
        nodes, scc = info['nodes'], info['scc']
        print(scc)
        scc.reverse()

        ltl = []
        for idx, node in enumerate(nodes[:-1]):
            if idx < len(nodes) - 1:
                next_node = nodes[idx+1]
                edge_idx = self.storage[node]['next'].index(next_node)
                edge = self.storage[node]['edges'][edge_idx]
                ltl.append(edge)
        
        if len(scc) > 1:
            scc_ltl = []
            entry = nodes[-1]
            entry_idx = scc.index(entry)
            scc = scc[entry_idx:] + scc[:entry_idx] + [entry]
            
            # build the ltl
            for idx, node in enumerate(scc):
                if idx < len(scc) - 1:
                    next_node = scc[idx+1 % len(scc)]
                    edge_idx = self.storage[node]['next'].index(next_node)
                    edge = self.storage[node]['edges'][edge_idx]
                    if not '1' in edge:
                        scc_ltl.append(edge)
            
            ltl = ltl + scc_ltl * FOREVER

        return ltl


def path_finding(formula):

    formula = reformat_ltl(formula)
    ltl_args = get_ltl_args(formula=formula)
    graph = gltl2ba(ltl_args)
    graph.save('test.png')
    #algo = SCCAlgorithm(graph=graph)
    #ltl = algo.search()
    
    algo = DijkstraAlgorithm(graph=graph, scc=['accept_S6', 'T0_S6', 'T0_S7', 'accept_S2', 'T2_S1', 'T2_S7', 'T1_S2'])
    algo.search()

    exit()
    return ltl_to_zones(ltl)


if __name__ == '__main__':

    f1 = '(!p U d) && (!e U (q && (!n U a)))'
    f2 = '<>b && a U b'
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
    
    formula = f9
    print('[INPUT FORMULA]', formula)
    
    goals, avoid_zones = path_finding(formula)
    print('[GOALS]', goals)
    print('[AVOID]', avoid_zones)
