# Implements the NetMF embeddings
# Based on: https://github.com/GemsLab/CONE-Align/blob/master/embedding.py
import random

import numpy as np
from scipy import sparse
from sparse_dot_mkl import dot_product_mkl
import aesara
from aesara import tensor as T
import networkx as nx
from typing import List, Union, Tuple
import pandas as pd
from .embedder import Embedder


class NetMFEmbedder(Embedder):

    def __init__(self, dim_embeddings, context_size, negative, normalize):
        self.indexdict = None
        self.emb_matrix = None

        self.dim_embeddings = dim_embeddings
        self.context_size = context_size
        self.negative = negative
        self.normalize = normalize

    # Full NMF matrix (which NMF factorizes with SVD)
    # Taken from MILE code
    def __netmf_mat_full(self, A, window=10, b=1.0):
        if not sparse.issparse(A):
            A = sparse.csr_matrix(A)
        # print "A shape", A.shape
        n = A.shape[0]
        vol = float(A.sum())
        L, d_rt = sparse.csgraph.laplacian(A, normed=True, return_diag=True)
        X = sparse.identity(n, format="csr") - L
        S = np.zeros_like(X)
        X_power = sparse.identity(n, format="csr")
        for i in range(window):
            # print "Compute matrix %d-th power" % (i + 1)
            X_power = dot_product_mkl(X_power, X)
            S += X_power
        S *= vol / window / b
        D_rt_inv = sparse.diags(d_rt ** -1)
        M = D_rt_inv.dot(D_rt_inv.dot(S).T)
        m = T.matrix()
        f = aesara.function([m], T.log(T.maximum(m, 1)))
        Y = f(M.todense().astype(aesara.config.floatX))
        return sparse.csr_matrix(Y)

    # Used in NetMF, AROPE
    def __svd_embed(self, prox_sim, dim):
        u, s, v = sparse.linalg.svds(prox_sim, dim, return_singular_vectors="u")
        return sparse.diags(np.sqrt(s)).dot(u.T).T

    def __netmf(self, A):
        prox_sim = self.__netmf_mat_full(A, self.context_size, self.negative)
        embed = self.__svd_embed(prox_sim, self.dim_embeddings)
        if self.normalize:
            norms = np.linalg.norm(embed, axis=1).reshape((embed.shape[0], 1))
            norms[norms == 0] = 1
            embed = embed / norms
        return embed

    def train(self, data: Union[np.ndarray, Union[str, List[Tuple[Union[int, float]]]]]):
        """
        Computes the NetMF Embeddings for the graph specified in data.
        :param data: The graph to encode. Either a string containing the path to an edgelist file, or an edgelist/
         numpy array of the form [(source, target, weight), (source, target, weight), ...]
        :return: Nothing
        """
        graph = None
        if type(data) == str:
            graph = nx.read_weighted_edgelist(data)
            adj = nx.adjacency_matrix(graph).todense().astype(float)

        elif type(data) in [list, np.ndarray]:
            graph = nx.from_pandas_edgelist(
                pd.DataFrame(data,
                             columns=["source", "target", "weight"]).astype({'source': 'int32', 'target': 'int32'}),
                                                     edge_attr=True)
            adj = nx.adjacency_matrix(graph).todense().astype(float)

        elif False: #Currently disabled
            # Efficiently creates an adjacency matrix without networkx.
            # Since we have an undirected graph, the adjacency matrix is symmetric along the diagonal.
            # Thus, two edges per node-pair must be present in our edgelist. This is done by duplicating the edges while
            # changing the source for the target, e.g. swapping columns 0 and 1.
            # This is most efficiently done via some fancy indexing,
            # see https://stackoverflow.com/questions/20265229/rearrange-columns-of-numpy-2d-array
            permutation = [1, 0, 2]
            idx = np.empty_like(permutation)
            idx[permutation] = np.arange(len(permutation))
            data = np.vstack([data, data[:, idx]])

            # https://stackoverflow.com/a/29148205
            shape = tuple((data.max(axis=0)[:2] + 1).astype(int))
            adj = sparse.coo_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape,
                                    dtype=data.dtype).todense()
            uid_inds = list(set(data[:, 0]))
            uid_inds.sort()
            del data

        else:
            raise Exception("Invalid data specified for NetMF computation")
        self.emb_matrix = self.__netmf(adj)
        # A dictionary mapping nodes IDs to rows in the embedding matrix
        if graph is not None:
            self.indexdict = dict(zip(list(graph.nodes()), range(graph.number_of_nodes())))
        else:
            self.indexdict = {int(uid): ind for ind, uid in enumerate(uid_inds)}

    def get_vectors(self, ordering: List[str] = None) -> np.ndarray:
        """
        Given an ordering (a list of node IDs), returns a numpy array storing their respective embeddings, as well as
        the ordering itself. The ordering of the array (row indices) is equivalent to the odering of the supplied list.
        If no ordering is specified, embeddings of all nodes are returned using the order of the keys of the index dict.
        :param ordering: An ordered list of node ids to retrieve the embeddings for
        :return: The embeddings and the ordering
        """
        if ordering is None:
            ordering = list(self.indexdict.keys())
            random.shuffle(ordering)
        embeddings = [self.emb_matrix[self.indexdict[k]] for k in ordering]
        embeddings = np.stack(embeddings, axis=0)
        return embeddings, ordering

    def get_vector(self, key: Union[int, str]) -> np.ndarray:
        """
        Returns the embedding for a given node ID.
        :param key: The node ID.
        :return: The embedding.
        """
        return self.emb_matrix[self.indexdict[key]]