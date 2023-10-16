import pickle
from algo.graph_utils import graph
import os 

graph_filename = './algo/convert/graph.pkl'

with open(graph_filename, 'rb') as f:
    g = pickle.load(f)

graph_conv = graph.Graph((0, 0), (0, 0), algo='Dijkstra')
print(g.startpos)
graph_conv.startpos = g.startpos
graph_conv.endpos = g.endpos
graph_conv.vertices = g.vertices
graph_conv.edges = g.edges
graph_conv.vex2idx = g.vex2idx
graph_conv.neighbors = g.neighbors
graph_conv.reward_list = g.reward_list
graph_conv.distance = g.distance
graph_conv.predecessor = g.predecessor
graph_conv.success = g.success
graph_conv.algo = g.algo

with open('./algo/convert/graph2.pkl', 'wb') as f:
            pickle.dump(graph_conv, f)