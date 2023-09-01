import pygraphviz as pgv
from subprocess import Popen, PIPE
import re
import sys
import __main__


# NOTE: assume no '||' operations
def ap_satisfied_check(formula):
    formula = formula.replace('(', '').replace(')', '').split(' ')
    while '' in formula:
        formula.remove('')
    while '&&' in formula:
        formula.remove('&&')
    if len(formula) > 1:
        pos_ap_num = 0
        for e in formula:
            if '!' not in e:
                pos_ap_num += 1
                if pos_ap_num > 1:
                    return False
    return True


class Graph:
    def __init__(self):
        self.graph = pgv.AGraph(directed=True, strict=False)
        self.accepting_nodes = []

    def title(self, title):
        self.graph.graph_attr['label'] = title

    def node(self, name, label, accepting=False):
        self.graph.add_node(name, label=name, shape='doublecircle' if accepting else 'circle')
        #self.graph.add_node(name, label=label, shape='doublecircle' if accepting else 'circle')
        if accepting:
            self.accepting_nodes.append(name)

    def edge(self, src, dst, label):
        if not '||' in label:
            if src == dst:
                return
            if not ap_satisfied_check(label):
                return
            else:
                self.graph.add_edge(src, dst, key=label, label=label, color='red')
        else:
            sub_formulas = label.split('||')
            for f in sub_formulas:
                if ap_satisfied_check(f):
                    self.graph.add_edge(src, dst, key=f, label=f, color='red')

    def simplify(self):
        for node in self.graph.iternodes():
            if not 'init' in node and len(list(self.graph.iterinedges(node))) == 0:
                self.graph.remove_node(node)
    
    def build_sub_graph(self, sub_graph_nodes):
        all_nodes = list(self.graph.iternodes())
        remove_nodes = [node for node in all_nodes if not node in sub_graph_nodes]
        for node in remove_nodes:
            self.remove_node(node)
        
        return self

    def save(self, path):
        self.graph.layout('dot')
        self.graph.draw(path)

    def save_dot(self, path):
        raise NotImplementedError

    def __str__(self):
        return str(self.graph)
    
    def nodes(self):
        return self.graph.nodes()
    
    def edges(self):
        return self.graph.edges()
    
    def iternodes(self):
        return self.graph.iternodes()
    
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
        graph = Graph()
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

        graph = Ltl2baParser.parse(match.group(1))
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
