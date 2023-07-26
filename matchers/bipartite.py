import numpy as np
from collections import deque
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
        eve_nodes = ["L_" + str(k) for k in eve_uids]
        alice_nodes = ["S_" + str(k) for k in alice_uids]
        bip_graph = nx.Graph()  # Add nodes with the node attribute "bipartite"
        bip_graph.add_nodes_from(eve_nodes, bipartite=0)
        bip_graph.add_nodes_from(alice_nodes, bipartite=1)
        # Add edges with weights
        for r in range(len(eve_uids)):
            for s in range(len(alice_uids)):
                bip_graph.add_edge("L_" + eve_uids[r], "S_" + alice_uids[s], weight=pw_dists[r][s])
        return bipartite.matching.minimum_weight_full_matching(bip_graph, None, "weight")


class GaleShapleyMatcher():
    def __init__(self, metric: str = "cosine", workers: int = -1):
        available_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
                             "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis",
                             "matching", "l1", "l2", "manhattan",
                             "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                             "sqeuclidean", "yule"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        self.metric = metric
        self.workers = workers

    def __gale_shapley(self, A, B, A_pref, B_rank):
        """Create a stable matching using the
        Gale-Shapley algorithm.

        A -- set[str].
        B -- set[str].
        A_pref -- dict[str, list[str]].
        B_pref -- dict[str, list[str]].

        Output: list of (a, b) pairs.
        """
        # B_rank = pref_to_rank(B_pref)
        ask_list = {a: deque(bs) for a, bs in A_pref.items()}
        pair = {}
        #
        remaining_A = set(A)
        while len(remaining_A) > 0:
            a = remaining_A.pop()
            b = ask_list[a].popleft()
            if b not in pair:
                pair[b] = a
            else:
                a0 = pair[b]
                b_prefer_a0 = B_rank[b][a0] < B_rank[b][a]
                if b_prefer_a0:
                    remaining_A.add(a)
                else:
                    remaining_A.add(a0)
                    pair[b] = a

        return pair
    def match(self, alice_data, alice_uids, eve_data, eve_uids):
        if len(alice_uids) <= len(eve_uids):
            smaller_data = alice_data
            larger_data = eve_data
            smaller_uids = alice_uids
            larger_uids = eve_uids
        else:
            smaller_data = eve_data
            larger_data = alice_data
            smaller_uids = eve_uids
            larger_uids = alice_uids

        pw_dists = pairwise_distances(smaller_data, larger_data, metric=self.metric, n_jobs=self.workers)
        smaller_nodes = ["S_" + str(k) for k in smaller_uids]
        larger_nodes = ["L_" + str(k) for k in larger_uids]
        smaller_pref = {}
        for smaller_ind in range(len(pw_dists)):
            smaller_pref["S_" + str(smaller_uids[smaller_ind])] = ["L_" + str(x) for _, x in sorted(
                zip(list(pw_dists[smaller_ind]), larger_uids))]

        larger_rank = {}
        for larger_ind in range(len(pw_dists[0])):
            tmp = {}
            for smaller_ind in range(len(pw_dists)):
                tmp["S_" + str(smaller_uids[smaller_ind])] = pw_dists[smaller_ind][larger_ind]
            larger_rank["L_" + str(larger_uids[larger_ind])] = tmp
        return self.__gale_shapley(A=set(smaller_nodes), B=set(larger_nodes), A_pref=smaller_pref, B_rank=larger_rank)


class SymmetricMatcher():
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
        pw_dists = pairwise_distances(alice_data, eve_data, metric=self.metric, n_jobs=self.workers)
        alice_pref = {}
        for alice_ind in range(len(pw_dists)):
            tmp = ["L_" + str(x) for _, x in sorted(zip(list(pw_dists[alice_ind]), eve_uids))]
            alice_pref["S_" + str(alice_uids[alice_ind])] = tmp[0]

        pw_dists = np.transpose(pw_dists)
        eve_pref = {}
        for eve_ind in range(len(pw_dists)):
            tmp = ["S_" + str(x) for _, x in sorted(zip(list(pw_dists[eve_ind]), alice_uids))]
            eve_pref["L_" + str(eve_uids[eve_ind])] = tmp[0]

        matching = {}
        for a, e in alice_pref.items():
            if eve_pref[e] == a:
                matching[e] = a
        return matching