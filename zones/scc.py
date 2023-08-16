from ltl_cmd import Graph


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

    def DFS_path(self, start, goal_scc=None):
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
        path = self.DFS_path(v)
        nodes_path = path['nodes']
        ltl_path = []
        for idx, node in enumerate(nodes_path[:-1]):
            if idx < len(nodes_path) - 1:
                next_node = nodes_path[idx+1]
                edge_idx = self.graph.storage[node]['next'].index(next_node)
                edge = self.graph.storage[node]['edges'][edge_idx]
                ltl_path.append(edge)
        path['ltl'] = ltl_path

        return path


if __name__ == '__main__':

    from types import SimpleNamespace
    from ltl_cmd import gltl2ba

    args = SimpleNamespace()
    args.formula = '(! w) U ( r && ((! y) U j))'
    args.formula = '(!p U d) && (!e U (q && (!n U a)))'
    args.formula = '(! w) U ( r && ((! y) U j)) U (! y)'
    args.formula = '<>b && a U b'
    args.formula = '[](<>(o && X (<> (c && X<> d))))'
    args.formula = 'GFa && GFb'
    args.formula = '[]<>a && []<>b'
    args.file = None

    args.s = False
    args.d = False
    args.l = False
    args.p = False
    args.o = False
    args.c = False
    args.a = False

    args.graph = False
    args.output_graph = open('ba', 'w')
    args.dot = False
    args.output_dot = open('ba.gv', 'w')

    graph = gltl2ba(args)
    algo = SCC_Algorithm(graph=graph)
    path = algo.search()
    print(path)
