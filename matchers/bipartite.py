import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances

from .matcher import Matcher


class MinWeightMatcher(Matcher):

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

        smaller_data = alice_data if len(alice_uids) < len(eve_uids) else eve_data
        larger_data = alice_data if len(alice_uids) >= len(eve_uids) else eve_data

        smaller_uids = alice_uids if len(alice_uids) < len(eve_uids) else eve_uids
        larger_uids = alice_uids if len(alice_uids) >= len(eve_uids) else eve_uids

        pw_sims = pairwise_distances(larger_data, smaller_data, metric=self.metric, n_jobs=self.workers)

        row_ind, col_ind = linear_sum_assignment(pw_sims)

        mapping = {}
        for larger, smaller in zip(row_ind, col_ind):
            mapping["S_"+str(smaller_uids[smaller])] = "L_"+str(larger_uids[larger])

        return mapping


class GaleShapleyMatcher(Matcher):
    # Partially based on
    # https://johnlekberg.com/blog/2020-08-22-stable-matching.html
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
        smaller_data = alice_data if len(alice_uids) < len(eve_uids) else eve_data
        larger_data = alice_data if len(alice_uids) >= len(eve_uids) else eve_data

        smaller_uids = alice_uids if len(alice_uids) < len(eve_uids) else eve_uids
        larger_uids = alice_uids if len(alice_uids) >= len(eve_uids) else eve_uids

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
        matching = self.__gale_shapley(A=set(smaller_nodes), B=set(larger_nodes), A_pref=smaller_pref, B_rank=larger_rank)
        # Swap keys and values to ensure consistency with other matchers
        return dict((v,k) for k,v in matching.items())


class SymmetricMatcher(Matcher):
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

        smaller_data = alice_data if len(alice_uids) < len(eve_uids) else eve_data
        larger_data = alice_data if len(alice_uids) >= len(eve_uids) else eve_data

        smaller_uids = alice_uids if len(alice_uids) < len(eve_uids) else eve_uids
        larger_uids = alice_uids if len(alice_uids) >= len(eve_uids) else eve_uids

        pw_dists = pairwise_distances(larger_data, smaller_data, metric=self.metric, n_jobs=self.workers)
        larger_pref = {}

        for larger_ind in range(pw_dists.shape[0]):
            tmp = ["S_" + str(x) for _, x in sorted(zip(list(pw_dists[larger_ind]), smaller_uids))]
            larger_pref["L_" + str(larger_uids[larger_ind])] = tmp[0]

        pw_dists = np.transpose(pw_dists)
        smaller_pref = {}
        for smaller_ind in range(pw_dists.shape[0]):
            tmp = ["L_" + str(x) for _, x in sorted(zip(list(pw_dists[smaller_ind]), larger_uids))]
            smaller_pref["S_" + str(smaller_uids[smaller_ind])] = tmp[0]

        matching = {}
        for s, l in smaller_pref.items():
            if larger_pref[l] == s:
                matching[s] = l
        return matching
