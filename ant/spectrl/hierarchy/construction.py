'''
Constructing task automaton and abstract reachability graph from spec.
'''
from copy import copy
from spectrl.main.spec_compiler import Cons, land
from spectrl.hierarchy.reachability import AbstractEdge, AbstractReachability


class TaskAutomaton:
    '''
    Task Automaton without registers.

    Parameters:
        delta: list of list of (int, predicate) pairs.
               predicate: state, resource -> float.
        final_states: set of int (final monitor states).

    Initial state is assumed to be 0.
    '''

    def __init__(self, delta, final_states):
        self.delta = delta
        self.final_states = final_states
        self.num_states = len(self.delta)


def automaton_graph_from_spec(spec):
    '''
    Constructs task automaton and abstract reachability graph from the specification.

    Parameters:
        spec: TaskSpec

    Returns:
        (automaton, abstract_reach): TaskAutomaton * AbstractReachability
    '''
    if spec.cons == Cons.ev:
        # Case 1: Objectives

        # Step 1a: Construct task automaton
        delta = [[(0, true_pred), (1, spec.predicate)], [(1, true_pred)]]
        automaton = TaskAutomaton(delta, set([1]))

        # Step 1b: Construct abstract graph
        abstract_graph = [[AbstractEdge(1, spec.predicate, [true_pred])],
                          [AbstractEdge(1, None, [true_pred])]]
        abstract_reach = AbstractReachability(abstract_graph, set([1]))

    elif spec.cons == Cons.alw:
        # Case 2: Constraints

        # Step 2a: Get automaton and graph for subtask
        a1, r1 = automaton_graph_from_spec(spec.subtasks[0])

        # Step 2b: Construct task automaton
        delta = [[(t, land(b, spec.predicate)) for t, b in trans] for trans in a1.delta]
        automaton = TaskAutomaton(delta, set(a1.final_states))

        # Step 2c: Construct abstract graph
        abstract_graph = []
        for edges in r1.abstract_graph:
            new_edges = []
            for edge in edges:
                if edge.predicate is not None:
                    new_predicate = land(edge.predicate, spec.predicate)
                else:
                    new_predicate = None
                new_constraints = [land(b, spec.predicate) for b in edge.constraints]
                new_edges.append(AbstractEdge(edge.target, new_predicate, new_constraints))
            abstract_graph.append(new_edges)
        abstract_reach = AbstractReachability(abstract_graph, set(r1.final_vertices))

    elif spec.cons == Cons.seq:
        # Case 3: Sequencing

        # Step 3a: Get automaton and graph for subtasks
        a1, r1 = automaton_graph_from_spec(spec.subtasks[0])
        a2, r2 = automaton_graph_from_spec(spec.subtasks[1])

        # Step 3b: Construct task automaton
        delta1 = [[(t, b) for t, b in trans] for trans in a1.delta]
        delta2 = [[(t + a1.num_states, b) for t, b in trans] for trans in a2.delta]

        delta = delta1 + delta2
        for s in a1.final_states:
            for t, b in a2.delta[0]:
                delta[s].append((t + a1.num_states, b))

        automaton = TaskAutomaton(delta, set([t + a1.num_states for t in a2.final_states]))

        # Step 3c: Construct abstract graph
        abstract_graph = [[copy_edge(e) for e in edges] for edges in r1.abstract_graph]

        for edges in r2.abstract_graph[1:]:
            new_edges = []
            for e in edges:
                new_target = e.target + r1.num_vertices - 1
                new_edges.append(AbstractEdge(new_target, e.predicate, copy(e.constraints)))
            abstract_graph.append(new_edges)

        for v in r1.final_vertices:
            abstract_graph[v] = []
            for e in r2.abstract_graph[0]:
                new_target = e.target + r1.num_vertices - 1
                new_constraints = r1.abstract_graph[v][0].constraints + e.constraints
                abstract_graph[v].append(AbstractEdge(new_target, e.predicate, new_constraints))

        final_vertices = set([t + r1.num_vertices - 1 for t in r2.final_vertices])
        abstract_reach = AbstractReachability(abstract_graph, final_vertices)

    elif spec.cons == Cons.choose:
        # Case 4: Choice

        # Step 4a: Get automaton and graph for subtasks
        a1, r1 = automaton_graph_from_spec(spec.subtasks[0])
        a2, r2 = automaton_graph_from_spec(spec.subtasks[1])

        # Step 4b: Construct task automaton
        delta01 = [(t + 1, b) for t, b in a1.delta[0]]
        delta02 = [(t + a1.num_states + 1, b) for t, b in a2.delta[0]]
        delta0 = [delta01 + delta02]
        delta1 = [[(t + 1, b) for t, b in trans] for trans in a1.delta]
        delta2 = [[(t + a1.num_states + 1, b) for t, b in trans] for trans in a2.delta]
        delta = delta0 + delta1 + delta2

        final_states_1 = [t + 1 for t in a1.final_states]
        final_states_2 = [t + a1.num_states + 1 for t in a2.final_states]
        final_states = set(final_states_1 + final_states_2)
        automaton = TaskAutomaton(delta, final_states)

        # Step 4c: Construct abstract graph
        abstract_graph = [[]]
        for e in r1.abstract_graph[0]:
            abstract_graph[0].append(copy_edge(e))
        for e in r2.abstract_graph[0]:
            new_target = e.target + r1.num_vertices - 1
            abstract_graph[0].append(AbstractEdge(new_target, e.predicate, copy(e.constraints)))

        for edges in r1.abstract_graph[1:]:
            new_edges = [copy_edge(e) for e in edges]
            abstract_graph.append(new_edges)

        for edges in r2.abstract_graph[1:]:
            new_edges = []
            for e in edges:
                new_target = e.target + r1.num_vertices - 1
                new_edges.append(AbstractEdge(new_target, e.predicate, copy(e.constraints)))
            abstract_graph.append(new_edges)

        final_vertices = r1.final_vertices.union(set(
            [t + r1.num_vertices - 1 for t in r2.final_vertices]))
        abstract_reach = AbstractReachability(abstract_graph, final_vertices)

    return automaton, abstract_reach


def true_pred(sys_state, res_state):
    return 1e9


def copy_edge(edge):
    return AbstractEdge(edge.target, edge.predicate, copy(edge.constraints))
