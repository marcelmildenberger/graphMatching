from sklearn.neighbors import NearestNeighbors
import numpy as np

class NNMatcher():
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

        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=self.metric, n_jobs=self.workers).fit(larger_data)
        distances, indices = nn.kneighbors(smaller_data)

        mapping = {}
        for i, nearest in enumerate(indices):
            mapping["S_"+smaller_uids[i]] = "L_"+larger_uids[nearest[0]]

        return mapping