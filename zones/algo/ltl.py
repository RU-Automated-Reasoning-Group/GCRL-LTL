import pygraphviz as pgv
from subprocess import Popen, PIPE
import re
import sys
import itertools
from collections import OrderedDict
import __main__


# NOTE: see spot.atomic_prop_collect() for robust implementation
def atomic_prop_collect(f):

    aps = str(f).replace('!', '').replace(' ', '').replace('(', '').replace(')', '').replace('G', '').replace('F', '').replace('&&', ' ').split()
    if '1' in aps:
        aps.remove('1')
    
    return aps

# NOTE: assumption: suppose all edges are in form of:
# (D1 ap1) ^ (D2 ap2) ^ ... ^ (Dn apn), where D denotes
# the pos/neg of an ap, and each ap presents only once.
# This function should also work for (1).
def edge_ap_check(f):
    aps = atomic_prop_collect(f)
    pos_ap_num, neg_ap_num = 0, 0
    aps_set = {'pos': [], 'neg': []}
    for ap in aps:
        if '!{}'.format(ap) not in str(f):
            pos_ap_num += 1
            aps_set['pos'].append(str(ap))
        else:
            neg_ap_num += 1
            aps_set['neg'].append(str(ap))
    assert pos_ap_num + neg_ap_num == len(aps)

    return pos_ap_num, neg_ap_num, aps_set


def blue_edge_satisfiable_check(f):
    pos_ap_num, neg_ap_num, aps_set = edge_ap_check(f)

    return True if pos_ap_num <= 1 else False


def red_edge_satisfiable_check(f, accepting):
    pos_ap_num, neg_ap_num, aps_set = edge_ap_check(f)

    if accepting:
        return True if pos_ap_num <= 1 else False
    else:
        return True if pos_ap_num == 0 else False


def blue_edge_simplify(f):
    f = f.replace(' ', '').replace('&&', ' && ')
    aps = atomic_prop_collect(f)
    if len(aps) == 0:
        return '(1)'
    elif len(aps) == 1:
        return '{}'.format(f)
    else:
        pos_ap_num, neg_ap_num, aps_set = edge_ap_check(f)
        if neg_ap_num > 1 and pos_ap_num == 0:
            return f
        elif pos_ap_num == 0:
            return '({})'.format(f)
        elif pos_ap_num == 1:
            return '({})'.format(aps_set['pos'][0])


class PathGraph:
    def __init__(self):
        self.graph = pgv.AGraph(directed=True, strict=False)
        self.accepting_nodes = []
        self.storage = {}

    def build(self, graph):
        
        # parse graph
        self.accepting_nodes = graph.accepting_nodes
        for node in graph.iternodes():
            self.storage[node] = OrderedDict({'in': [], 'out': [], 'self': '{}'})
            for edge in graph.iterinedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ').replace('(', '').replace(')', '')
                if src != dst:  # this is required
                    self.storage[node]['in'].append((src, f))
            for edge in graph.iteroutedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ').replace('(', '').replace(')', '')
                if src != dst:
                    self.storage[node]['out'].append((dst, f))
                else:
                    self.storage[node]['self'] = f
        
        # build nodes
        node_name_dict = {}
        
        # init node
        for node in self.storage:
            if 'init' in node:
                init_node = node
                name = '{}|empty'.format(init_node)
                node_name_dict[init_node] = [name]
                self.node(name=name, label=name, accepting=True if node in self.accepting_nodes else False)
        
        # other nodes
        for node in self.storage:
            if not node in node_name_dict.keys():
                node_name_dict[node] = []
            if self.storage[node]['in']:
                for in_node, in_f in self.storage[node]['in']:
                    name = '{}|{}'.format(node, in_f)
                    self.node(name=name, label=name, accepting=True if node in self.accepting_nodes else False)
                    if name not in node_name_dict[node]:
                        node_name_dict[node].append(name)

        # build edges
        for edge in graph.iteredges():
            src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ').replace('(', '').replace(')', '')
            if src != dst:
                dst_name = '{}|{}'.format(dst, f)
                src_node_names = node_name_dict[src]
                for src_name in src_node_names:
                    assert self.has_node(src_name) and self.has_node(dst_name), 'src and dst nodes must be in the path finding graph'
                    label = self.storage[src]['self']
                    self.edge(src=src_name, dst=dst_name, label=label)

        # simplify
        nodes_to_remove = []
        for node in self.iternodes():
            if node.split('|')[1] == '1' or '!' in node:

                in_edges = list(self.iterinedges(node))
                out_edges = list(self.iteroutedges(node))

                for in_edge, out_edge in itertools.product(in_edges, out_edges):
                    if in_edge[0] in self.accepting_nodes and out_edge[1] in self.accepting_nodes:
                        acc_circle_f = in_edge.attr['label']
                        new_name = node.split('|')[0] + '|' + acc_circle_f
                        node.attr.update(string=new_name, name=new_name, label=new_name)
                    else:
                        self.edge(src=in_edge[0], dst=out_edge[1], label=out_edge.attr['label'])
                        if node not in nodes_to_remove:
                            nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.remove_node(node)

        # build title
        self.title(graph.graph.graph_attr['label'])

    def title(self, title):
        self.graph.graph_attr['label'] = title

    def node(self, name, label, accepting=False):
        self.graph.add_node(name, name=name, label=label, shape='doublecircle' if accepting else 'circle')
        if accepting:
            self.accepting_nodes.append(name)

    def edges(self):
        return self.graph.edges()
    
    def iternodes(self):
        return self.graph.iternodes()

    def iteredges(self, *args):
        return self.graph.iteredges(*args)
    
    def iterinedges(self, *args):
        return self.graph.iterinedges(*args)
    
    def iteroutedges(self, *args):
        return self.graph.iteroutedges(*args)
    
    def has_node(self, *args):
        return self.graph.has_node(*args)

    def remove_node(self, *args):
        self.graph.remove_node(*args)

    def edge(self, src, dst, label):
        self.graph.add_edge(src, dst, key=label, label=label, color='red')

    def get_edge(self, *args):
        return self.graph.get_edge(*args)

    def save(self, path):
        self.graph.layout('dot')
        self.graph.draw(path)

    def build_sub_graph(self, sub_graph_nodes):
        all_nodes = list(self.graph.iternodes())
        remove_nodes = [node for node in all_nodes if not node in sub_graph_nodes]
        for node in remove_nodes:
            self.remove_node(node)
        
        return self

    def __str__(self):
        return str(self.graph)


class BuchiGraph:
    def __init__(self):
        self.graph = pgv.AGraph(directed=True, strict=False)
        self.accepting_nodes = []
        self.impossible_nodes = []

    def title(self, title):
        self.graph.graph_attr['label'] = title

    def node(self, name, label, accepting=False):
        self.graph.add_node(name, name=name, label=name, shape='doublecircle' if accepting else 'circle')
        if accepting:
            self.accepting_nodes.append(name)

    def edge(self, src, dst, label):
        if not '||' in label:
            # red edge
            if src == dst:
                accepting = src in self.accepting_nodes
                if accepting and not red_edge_satisfiable_check(label, accepting=True):
                    return
                elif not accepting and not red_edge_satisfiable_check(label, accepting=False):
                    # NOTE: handle failure properly
                    #assert False, 'red_edge_satisfiable_check failed on [{}]->({})-[{}]'.format(src, label, dst)
                    self.impossible_nodes.append(src)
                    return
                elif label == '(1)':
                    return
                else:
                    self.graph.add_edge(src, dst, key=label, label=label, color='red')
            # blue edge
            elif src != dst:
                if not blue_edge_satisfiable_check(label):
                    return
                else:
                    f = blue_edge_simplify(label)
                    self.graph.add_edge(src, dst, key=f, label=f, color='blue')
        else:
            sub_formulas = label.split('||')
            for f in sub_formulas:
                self.edge(src, dst, label=f)

    def simplify(self):

        edges_to_remove = []
        for node in self.impossible_nodes:
            for edge in self.iteroutedges(node):
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            self.remove_edge(edge)

        nodes_to_remove = []
        for node in self.iternodes():
            if not 'init' in node:
                in_edges = list(self.iterinedges(node))
                if len(in_edges) == 0:
                    nodes_to_remove.append(node)
                elif len(in_edges) == 1:
                    src, dst = in_edges[0]
                    if src == dst:
                        nodes_to_remove.append(node)
        for node in nodes_to_remove:
            self.remove_node(node)

        direct_trans = {}
        for node in self.iternodes():
            direct_trans[node] = []
            for edge in self.iteroutedges(node):
                src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ').replace('(', '').replace(')', '')
                if src != dst and (f == '1' or '!' in f):
                    direct_trans[str(node)].append(f)
        
        edges_to_remove = []
        for node in direct_trans:
            if direct_trans[node]:
                for edge in self.iteroutedges(node):
                    src, dst, f = edge[0], edge[1], edge.attr['label'].replace(' ', '').replace('&&', ' && ').replace('(', '').replace(')', '')
                    if src != dst and f not in direct_trans[node]:
                        edges_to_remove.append(edge)

        for edge in edges_to_remove:
           self.remove_edge(edge)

    def save(self, path):
        self.graph.layout('dot')
        self.graph.draw(path)

    def __str__(self):
        return str(self.graph)
    
    def nodes(self):
        return self.graph.nodes()
    
    def edges(self):
        return self.graph.edges()
    
    def iternodes(self):
        return self.graph.iternodes()

    def iteredges(self, *args):
        return self.graph.iteredges(*args)
    
    def iterinedges(self, *args):
        return self.graph.iterinedges(*args)
    
    def iteroutedges(self, *args):
        return self.graph.iteroutedges(*args)

    def remove_edge(self, *args):
        self.graph.remove_edge(*args)

    def remove_node(self, *args):
        self.graph.remove_node(*args)

#
# parser for ltl2ba output
#


class Ltl2baParser:
    prog_title = re.compile('^never\s+{\s+/\* (.+?) \*/$')
    prog_node = re.compile('^([^_]+?)_([^_]+?):$')
    prog_edge = re.compile('^\s+:: (.+?) -> goto (.+?)$')
    prog_skip = re.compile('^\s+(?:skip)$')
    prog_ignore = re.compile('(?:^\s+do)|(?:^\s+if)|(?:^\s+od)|'
                             '(?:^\s+fi)|(?:})|(?:^\s+false);?$')

    @staticmethod
    def parse(ltl2ba_output, ignore_title=True):
        graph = BuchiGraph()
        src_node = None
        for line in ltl2ba_output.split('\n'):
            if Ltl2baParser.is_title(line):
                title = Ltl2baParser.get_title(line)
                if not ignore_title:
                    graph.title(title)
            elif Ltl2baParser.is_node(line):
                name, label, accepting = Ltl2baParser.get_node(line)
                graph.node(name, label, accepting)
                src_node = name
            elif Ltl2baParser.is_edge(line):
                dst_node, label = Ltl2baParser.get_edge(line)
                assert src_node is not None
                graph.edge(src_node, dst_node, label)
            elif Ltl2baParser.is_skip(line):
                assert src_node is not None
                graph.edge(src_node, src_node, "(1)")
            elif Ltl2baParser.is_ignore(line):
                pass
            else:
                print("--{}--".format(line))
                raise ValueError("{}: invalid input:\n{}"
                                 .format(Ltl2baParser.__name__, line))

        return graph

    @staticmethod
    def is_title(line):
        return Ltl2baParser.prog_title.match(line) is not None

    @staticmethod
    def get_title(line):
        assert Ltl2baParser.is_title(line)
        return Ltl2baParser.prog_title.search(line).group(1)

    @staticmethod
    def is_node(line):
        return Ltl2baParser.prog_node.match(line) is not None

    @staticmethod
    def get_node(line):
        assert Ltl2baParser.is_node(line)
        prefix, label = Ltl2baParser.prog_node.search(line).groups()
        return (prefix + "_" + label, label,
                True if prefix == "accept" else False)

    @staticmethod
    def is_edge(line):
        return Ltl2baParser.prog_edge.match(line) is not None

    @staticmethod
    def get_edge(line):
        assert Ltl2baParser.is_edge(line)
        label, dst_node = Ltl2baParser.prog_edge.search(line).groups()
        return (dst_node, label)

    @staticmethod
    def is_skip(line):
        return Ltl2baParser.prog_skip.match(line) is not None

    @staticmethod
    def is_ignore(line):
        return Ltl2baParser.prog_ignore.match(line) is not None


def gltl2ba(args):
    
    ltl = get_ltl_formula(args.file, args.formula)
    (output, err, exit_code) = run_ltl2ba(args, ltl)

    if exit_code != 1:

        prog = re.compile("^[\s\S\w\W]*?"
                            "(never\s+{[\s\S\w\W]+?})"
                            "[\s\S\w\W]+$")
        match = prog.search(output)
        assert match, output

        graph = Ltl2baParser.parse(match.group(1), ignore_title=False)
        graph.simplify()

    return graph


def get_ltl_formula(file, formula):
    assert file is not None or formula is not None
    if file:
        try:
            ltl = file.read()
        except Exception as e:
            eprint("{}: {}".format(__main__.__file__, str(e)))
            sys.exit(1)
    else:
        ltl = formula
    ltl = re.sub('\s+', ' ', ltl)
    if len(ltl) == 0 or ltl == ' ':
        eprint("{}: empty ltl formula.".format(__main__.__file__))
        sys.exit(1)
    return ltl


def run_ltl2ba(args, ltl):
    ltl2ba_args = ["ltl2ba", "-f", ltl]

    ltl2ba_args += list("-{}".format(x) for x in "dslpoca"
                        if getattr(args, x))

    try:
        process = Popen(ltl2ba_args, stdout=PIPE)
        (output, err) = process.communicate()
        exit_code = process.wait()
    except FileNotFoundError as e:
        eprint("{}: ltl2ba not found.\n".format(__main__.__file__))
        eprint("Please download ltl2ba from\n")
        eprint("\thttp://www.lsv.fr/~gastin/ltl2ba/ltl2ba-1.2b1.tar.gz\n")
        eprint("compile the sources and add the binary to your $PATH, e.g.\n")
        eprint("\t~$ export PATH=$PATH:path-to-ltlb2ba-dir\n")
        sys.exit(1)

    output = output.decode('utf-8')

    return output, err, exit_code


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
