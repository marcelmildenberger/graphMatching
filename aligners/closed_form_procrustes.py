# Implements a solution to the closed form procrustes problem, i.e. aligning vector spaces with parallel data as
# ground truth. Will be used for evaluation purposes only.

import numpy as np

def normalized(a, axis=-1, order=2):
    # https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class ProcrustesAligner:

    def __init__(self):
        pass

    def align(self, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """
        Solves the Orthogonal Procrustes Problem for two given matrices and returns a transformation matrix T.
        When multiplied withT, the source embeddings will be aligned to the target embeddings.
        Source and target matrices must be parallel, i.e. the embeddings of the same node/word must occur at the same
        index in both matrices.
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        :param src: Source embeddings (i.e. the ones to align).
        :param tgt: Target embeddings (the ones the source embeddings will be aligned to)
        :return: Transformation matrix.
        """
        assert src.shape == tgt.shape, "Source and target must have the same dimensionality"
        src = normalized(src)
        tgt = normalized(tgt)

        matrix_product = np.matmul(src.transpose(), tgt)
        U, s, V = np.linalg.svd(matrix_product)
        return np.matmul(U, V)
