import numpy as np
from scipy import sparse
from sparse_dot_mkl import dot_product_mkl
import aesara
from aesara import tensor as T
import networkx as nx
from typing import List, Union


class NetMFEmbedder():

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

    def train(self, data_dir: str):
        graph = nx.read_weighted_edgelist(data_dir)
        adj = nx.adjacency_matrix(graph).todense().astype(float)
        self.emb_matrix = self.__netmf(adj)
        self.indexdict = dict(zip(list(graph.nodes()), range(graph.number_of_nodes())))

    def get_vectors(self, ordering: List[str] = None) -> np.ndarray:
        if ordering is None:
            return self.emb_matrix, list(self.indexdict.keys())
        embeddings = [self.emb_matrix[self.indexdict[k]] for k in ordering]
        embeddings = np.stack(embeddings, axis=0)
        return embeddings, ordering

    def get_vector(self, key: Union[int, str]) -> np.ndarray:
        return self.emb_matrix[self.indexdict[key]]