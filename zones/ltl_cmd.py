import spot
from graphviz import Digraph
from subprocess import Popen, PIPE
import re
import argparse
import sys
from collections import OrderedDict
import __main__


class Graph:
    def __init__(self, never_claim=None):
        self.dot = Digraph()
        self.never_claim = never_claim  # never claim representation
        
        self.storage = OrderedDict()
        self.accepting_nodes = []
        self.label_to_name = {}
        self.name_to_label = {}
        self.not_count = 0
        # NOTE: maybe we need to
        # record self transition edges

    def title(self, str):
        self.dot.graph_attr.update(label=str)

    def node(self, name, label, accepting=False):
        num_peripheries = '2' if accepting else '1'
        self.dot.node(name, label, shape='circle', peripheries=num_peripheries)
        if accepting:
            self.accepting_nodes.append(name)
        self.storage[name] = OrderedDict({'next': [], 'edges': []})
        self.label_to_name[label] = name
        self.name_to_label[name] = label

    def edge(self, src, dst, label):
        if self.edge_check(label):
            self.dot.edge(src, dst, label)
            self.storage[src]['next'].append(dst)
            self.storage[src]['edges'].append(label)

    def edge_check(self, label):

        f = spot.formula(label)
        aut = spot.translate(f)
        ap = aut.ap()
        num_ap = len(ap)

        if num_ap == 0:  # '1'
            return False
        elif num_ap == 1:  # single ap
            return True
        else:
            return True

        # NOTE: following functions can be sub by simplify_never_claim()
        # else:                
        #     self.not_count = 0
        #     def count_not_op(_f):
        #         if _f._is(spot.op_Not):
        #             self.not_count += 1
            
        #     f.traverse(count_not_op)

        # return True if self.not_count >= num_ap - 1 else False

    def show(self):
        self.dot.render(view=True)

    def save_render(self, path, on_screen):
        self.dot.render(path, view=on_screen, cleanup=True)

    def save_dot(self, path):
        self.dot.save(path)

    def __str__(self):
        return str(self.dot)

    def reachable_check(self):

        visited = []
        def DFS(v):
            if v not in visited:
                visited.append(v)
                for next_v in self.storage[v]['next']:
                    DFS(next_v)

        start_v = None
        for v in self.storage.keys():
            if 'init' in v:  # naming convention
                start_v = v
                break

        DFS(start_v)

        never_str = self.never_claim.split('\n')[0]
        graph_str = '\n'.join(self.never_claim.split('\n')[1:])

        skip_idx = graph_str.find('skip') + 4
        if skip_idx:
            graph_str = graph_str[:skip_idx] + ';' + graph_str[skip_idx:]
            # NOTE: question, multiple 'skip'?
        graph_str = graph_str.split(';\n')

        for idx, s in enumerate(graph_str):
            if len(s) == 1:  # '}'
                continue
            node_info = s.split('\n')[0][:-1]
            if not node_info in visited:
                graph_str[idx] = ''
            else:
                if not 'skip' in graph_str[idx]:
                    graph_str[idx] += ';\n'
                else:
                    graph_str[idx] += '\n'

        output_str = never_str + '\n' + ''.join(graph_str)
        return output_str


class Ltl2baParser:
    prog_title = re.compile('^never\s+{\s+/\* (.+?) \*/$')
    prog_node = re.compile('^([^_]+?)_([^_]+?):$')
    prog_edge = re.compile('^\s+:: (.+?) -> goto (.+?)$')
    prog_skip = re.compile('^\s+(?:skip)$')
    prog_ignore = re.compile('(?:^\s+do)|(?:^\s+if)|(?:^\s+od)|'
                             '(?:^\s+fi)|(?:})|(?:^\s+false);?$')

    @staticmethod
    def parse(ltl2ba_output, ignore_title=True):
        #print(ltl2ba_output)
        graph = Graph(never_claim=ltl2ba_output)
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


def simplify_never_claim(input_str):

    graph_str = input_str.split('\n')
    for idx, s in enumerate(graph_str):
        requires_newline = True
        if '::' in graph_str[idx]:
            left_br_idx = s.find('(') 
            right_br_idx = len(s) - s[::-1].find(')')
            f = spot.formula(s[left_br_idx:right_br_idx])
            aut = spot.translate(f)
            ap = aut.ap()
            num_ap = len(ap)
            num_not = s.count('!')
            check_pass = num_not >= num_ap - 1
            if not check_pass:
                graph_str[idx] = ''
                requires_newline = False

        if requires_newline:
            graph_str[idx] += '\n'

    # NOTE: we should keep track of node names
    node_names = {}
    node_idx = 0
    for s in graph_str:
        if ('T' in s and 'goto' not in s) or ('accept' in s and 'goto' not in s):
            name = s[:-2]
            if node_idx == 0:
                node_names[name] = name.split('_')[0] + '_init'
            else:
                node_names[name] = name.split('_')[0] + '_' + str(node_idx)
            node_idx += 1

    graph_str = ''.join(graph_str)
    for original_name in node_names:
        graph_str = graph_str.replace(original_name, node_names[original_name])

    return graph_str


def remove_self_transition(input_str):

    header = input_str.split('*/')[0] + '*/'
    body = input_str.split('*/')[1].split('}')[0]
    nodes_info = body.split(';')[:-1]
    end = body.split(';')[-1] + '}'

    graph_str = header + '\n'
    for s in nodes_info:
        
        node_name = s.split(':')[0][1:]  # e.g., 'T0_init'

        transitions = []
        for line in s.split('\n'):
            if '->' in line:
                transitions.append(line.strip().split('::')[1].strip())
        
        self_transition_formula = None
        _transitions = []
        for trans in transitions:
            if node_name in trans:
                # NOTE: we assume no more than one self transition formula
                self_transition_formula = trans.split('->')[0].strip()
            else:
                _transitions.append(trans)

        transitions = []
        for trans in _transitions:
            
            f = spot.formula(trans.split('->')[0])
            dest = trans.split('->')[1]

            if self_transition_formula:
                self_f = spot.formula(self_transition_formula)
                new_f = spot.formula(str(f) + ' & ' + str(self_f)).simplify()
                transitions.append('(' + str(new_f) + ') ->' + dest)
            else:
                transitions.append('(' + str(f) + ') ->' + dest)
        
        transition_str = '\tif\n\t'
        for trans in transitions:
            transition_str += ':: '
            transition_str += trans
            transition_str += '\n\t'
        
        transition_str = transition_str[:-2]
        transition_str += '\n\tfi;\n'

        node_string = node_name + ':\n' + transition_str
        graph_str += node_string

    graph_str = graph_str[:-1] + end + '\n'
    return graph_str


def never_to_graph(never_claim):

    prog = re.compile("^[\s\S\w\W]*?"
                            "(never\s+{[\s\S\w\W]+?})"
                            "[\s\S\w\W]+$")
    match = prog.search(never_claim + '\n')
    assert match, never_claim

    graph = Ltl2baParser.parse(match.group(1))

    return graph


def gltl2ba(args):
    
    ltl = get_ltl_formula(args.file, args.formula)

    (output, err, exit_code) = run_ltl2ba(args, ltl)
    output = simplify_never_claim(output)
    
    if exit_code != 1:

        graph = never_to_graph(never_claim=output)
        output = graph.reachable_check()
        output = simplify_never_claim(output)
        output = remove_self_transition(output)
        graph = never_to_graph(never_claim=output)

        if args.output_graph is not None:
            graph.save_render(args.output_graph.name, args.graph)
            args.output_graph.close()
        elif args.graph:
            graph.show()

        if args.output_dot is not None:
            graph.save_dot(args.output_dot.name)
            args.output_dot.close()
        if args.dot:
            print(graph)

    else:
        eprint("{}: ltl2ba error:".format(__main__.__file__))
        eprint(output)
        sys.exit(exit_code)

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
