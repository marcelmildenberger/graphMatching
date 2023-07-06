import numpy as np

# Implements a solution to the closed form procrustes problem, i.e. aligning vector spaces with parallel data as
# ground truth. Will be used for evaluation purposes only.

def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

class ProcrustesAligner:

    def __init__(self):
        pass

    def align(self, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        assert src.shape == tgt.shape, "Source and target must have the same dimensionality"
        src = normalized(src)
        tgt = normalized(tgt)

        matrix_product = np.matmul(src.transpose(), tgt)
        U, s, V = np.linalg.svd(matrix_product)
        return np.matmul(U, V)

