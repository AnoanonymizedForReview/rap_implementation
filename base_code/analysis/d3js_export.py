import numpy as np
import networkx as nx
import json
from base_code.analysis.overall_clusters import ChainAggregator, ChainAggregatorMethod
from networkx.readwrite import json_graph


class Position:
    def __init__(self, x, y, fixed=True):
        self.x = x
        self.y = y
        self.fixed = fixed

class Dumper:
    '''A class for D3Js based visualization'''
    def __init__(self, **kwargs):
        self.H = kwargs['H']
        self.n = self.H.shape[0]
        self.cls = ChainAggregator(H=self.H, method=ChainAggregatorMethod.SYMMETRIC_H).get_clusters()
        self.exemplars = np.where(np.diagonal(self.H)==1)[0]
        self.inv_cls = dict((v, k) for k in self.cls for v in self.cls[k])
        self.S = kwargs.get('S', None)
        self.labels = kwargs.get('labels', range(self.n)) # takes as input a list of N labels, or chooses np.range(N) as labels
        self.positions = kwargs.get('positions', None) # type: list[Position] #optional, takes an N dimensional list defining position of data points.
        self.file_name = kwargs.get('file_name', 'dump.json') # output file name. Better leave it as it is, or you will need to change that in visualize/graph_functions.js
        self.extra = kwargs.get('extra', None) # attach some extra info if you want to edit graph_functions.js and access this info

    def dump(self, labels=None, positions=None, edge_label=None, extra=None):
        H_cpy = self.H.copy()
        np.fill_diagonal(H_cpy, 0)
        H_cpy = H_cpy.T
        rows, cols = np.where(H_cpy == 1)
        W = self.S

        #declare a multi-graph, add nodes and edges to it
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(self.n))
        edges = [(r, c, {'weight': W[r, c]}) for r, c in zip(rows.tolist(), cols.tolist())]
        G.add_edges_from(edges)

        for ix, deg in G.degree_iter():
            G.node[ix]['degree'] = deg # add degree to every node
            G.node[ix]['color'] = self.inv_cls[ix] # add color to node. Right now, color is based upon the cluster it is in
            G.node[ix]['is_exemplar'] = ix in self.exemplars # for additional info regarding exemplars

            G.node[ix]['neighbours'] = list(set( # neighbours of every point
                np.where(H_cpy[ix, :])[0].tolist()
            ).union(set(
                np.where(H_cpy[:, ix])[0].tolist())
            ))

            if self.labels is not None: # assign labels
                G.node[ix]['name'] = self.labels[ix]

            if self.extra is not None: # assign extra info if available
                G.node[ix]['extra'] = self.extra[ix]

            if self.positions is not None: # assign position list if available
                G.node[ix]['x'] = self.positions[ix].x
                G.node[ix]['y'] = self.positions[ix].y
                G.node[ix]['fixed'] = self.positions[ix].fixed

        #dump to json
        data = json_graph.node_link_data(G)
        with open(self.file_name, 'w') as f:
            json.dump(data, f, indent=4)
