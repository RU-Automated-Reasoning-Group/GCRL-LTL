from types import SimpleNamespace
from collections import OrderedDict

import spot
import pygraphviz as pgv

from ltl import gltl2ba
from ltl_progression import progress, _get_spot_format


FOREVER = 100


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
    ltl = progress(formula, '')
    ltl_spot = _get_spot_format(ltl)
    f = spot.formula(ltl_spot)
    f = spot.simplify(f)
    f = str(f).replace('&', '&&').replace('"', '').replace('|', '||').lower()
    f = f.replace('u', 'U').replace('f', '<>').replace('g', '[]').replace('x', 'X')

    return f


def ltl_to_zones(ltl, translation_function=None):

    if translation_function is None:
        def translate(word):
            return word
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


class AlgorithmGraph:
    def __init__(self, graph):
        self.storage = OrderedDict()
        self.accepting_nodes = graph.accepting_nodes
    
        for node in graph.iternodes():
            self.storage[node] = OrderedDict({'next': [], 'edges': []})
            for edge in graph.iteroutedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ')
                self.storage[src]['next'].append(dst)
                self.storage[src]['edges'].append(f)


class SCC_Algorithm:
    def __init__(self, graph):
        self.graph = graph
        self.graph_keys = self.graph.storage.keys()
        self.reset()

    def reset(self):
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
        for next_v in self.graph.storage[v]['next']:
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
                if x in self.graph.accepting_nodes:
                    accepting = True
                scc.append(x)
                if x == v:
                    self.SCCS.append([scc, accepting])
                    break

    def DFS_search(self, start, goal_scc=None):
        for scc in self.SCCS:
            if scc[1]:
                goal_scc = scc[0]
        stack = [(start, [start])]
        visited = set()
        while stack:
            (vertex, path) = stack.pop()
            if vertex not in visited:
                if vertex in goal_scc:
                    return {'nodes': path, 'scc': goal_scc}
                visited.add(vertex)
                for neighbor in self.graph.storage[vertex]['next']:
                    stack.append((neighbor, path + [neighbor]))
        
    def search(self, v=None):
        if v is None:
            for name in self.graph_keys:
                if 'init' in name:
                    v = name
        self.SCC_search(v)
        info = self.DFS_search(v)
        nodes, scc = info['nodes'], info['scc']
        scc.reverse()
        
        ltl = []
        if len(scc) == 1:
            for idx, node in enumerate(nodes[:-1]):
                if idx < len(nodes) - 1:
                    next_node = nodes[idx+1]
                    edge_idx = self.graph.storage[node]['next'].index(next_node)
                    edge = self.graph.storage[node]['edges'][edge_idx]
                    ltl.append(edge)
        else:
            # find the entry point to the scc
            entry = None
            for next_node in self.graph.storage[nodes[-1]]['next']:
                if next_node in scc:
                    entry = next_node
            nodes.append(entry)
            
            # re-positioning scc
            for idx, node in enumerate(nodes[:-1]):
                if idx < len(nodes) - 1:
                    next_node = nodes[idx+1]
                    edge_idx = self.graph.storage[node]['next'].index(next_node)
                    edge = self.graph.storage[node]['edges'][edge_idx]
                    ltl.append(edge)
            scc_ltl = []
            entry_idx = scc.index(entry)
            scc = scc[entry_idx:] + scc[:entry_idx] + [entry]
            
            # build the ltl
            for idx, node in enumerate(scc):
                if idx < len(scc) - 1:
                    next_node = scc[idx+1 % len(scc)]
                    edge_idx = self.graph.storage[node]['next'].index(next_node)
                    edge = self.graph.storage[node]['edges'][edge_idx]
                    if not '1' in edge:
                        scc_ltl.append(edge)

            ltl = ltl + scc_ltl * FOREVER

        return ltl


def path_finding(formula):

    formula = reformat_ltl(formula)
    print(formula)
    exit()
    ltl_args = get_ltl_args(formula=formula)
    graph = gltl2ba(ltl_args)
    graph.save('test.png')
    algo_graph = AlgorithmGraph(graph=graph)
    algo = SCC_Algorithm(graph=algo_graph)
    ltl = algo.search()
    
    return ltl_to_zones(ltl)


if __name__ == '__main__':

    # NOTE: good graphs    
    formula = '(!p U d) && (!e U (q && (!n U a)))'
    formula = '<>b && a U b'
    formula = '!j U (w && (!y U r))'
    formula = 'Fa'
    formula = 'GFa'
    formula = 'GFa && GFb'
    formula = '[]<>a && []<>b'

    # NOTE: should handle || correctly
    formula = '<>((b || q) && <>((e || p) && <>m))'  # NOTE: not the shortest path, check the graph again
    formula = '<>((c || n) && <>(r && <>d)) && <>(q && <>((r || t) && <>m))'  # NOTE: graph is wrong
    
    formula = '(! w) U ( r && ((! y) U j)) U (! y)'  # NOTE: graph is wrong, absolutely wrong
    formula = '[](<>(o && X (<> (c && X<> d))))'  # NOTE: graph is wrong, very strange graph

    # NOTE: graph is wrong, but the task is wrong too...
    # it is better we can handle this directly
    formula = '(! w) U ( r && ((! y) U j))'
    
    # DEBUG
    formula = 'GF(r && XF y) && G(!w)'
    print('[INPUT FORMULA]', formula)
    
    goals, avoid_zones = path_finding(formula)
    #print('[GOALS]', goals)
    #print('[AVOID]', avoid_zones)
    
    