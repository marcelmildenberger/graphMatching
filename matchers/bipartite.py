import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import pairwise_distances
from networkx.algorithms import bipartite

class MinWeightMatcher():

    def __init__(self, metric: str = "cosine", workers: int = -1):
        available_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
                             "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis",
                             "matching", "l1", "l2", "manhattan",
                             "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                             "sqeuclidean", "yule"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        self.metric = metric
        self.workers = workers

    def match(self, x, y):
        pw_dists = pairwise_distances(x, y, metric=self.metric, n_jobs=self.workers)
        x_nodes = ["X_" + str(k) for k in range(len(x))]
        y_nodes = ["Y_" + str(k) for k in range(len(y))]
        bip_graph = nx.Graph()  # Add nodes with the node attribute "bipartite"
        bip_graph.add_nodes_from(x_nodes, bipartite=0)
        bip_graph.add_nodes_from(y_nodes, bipartite=1)
        # Add edges with weights
        for r in range(len(pw_dists)):
            for s in range(len(pw_dists[r])):
                bip_graph.add_edge("X_" + str(r), "Y_" + str(s), weight=pw_dists[r][s])
        return bipartite.matching.minimum_weight_full_matching(bip_graph, None, "weight")