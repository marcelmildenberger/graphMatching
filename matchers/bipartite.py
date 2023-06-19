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

    def match(self, alice_data, alice_uids, eve_data, eve_uids):
        pw_dists = pairwise_distances(eve_data, alice_data, metric=self.metric, n_jobs=self.workers)
        eve_nodes = ["E_" + str(k) for k in eve_uids]
        alice_nodes = ["A_" + str(k) for k in alice_uids]
        bip_graph = nx.Graph()  # Add nodes with the node attribute "bipartite"
        bip_graph.add_nodes_from(eve_nodes, bipartite=0)
        bip_graph.add_nodes_from(alice_nodes, bipartite=1)
        # Add edges with weights
        for r in range(len(eve_uids)):
            for s in range(len(alice_uids)):
                bip_graph.add_edge("E_" + eve_uids[r], "A_" + alice_uids[s], weight=pw_dists[r][s])
        return bipartite.matching.minimum_weight_full_matching(bip_graph, None, "weight")