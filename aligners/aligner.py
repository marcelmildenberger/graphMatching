# Specifies an interface that must be implemented by aligners in order to be used in the graph matching attack
from abc import ABC, abstractmethod

import numpy as np


class Aligner(ABC):

    @abstractmethod
    def align(self, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """
        Computes an alignment matrix that maps the source embeddings (src) to the target embedding space (tgt).
        :param src: A numpy array containing the source embedding space. One embedding per row. Number of columns is equal to embedding dimensionality.
        :param tgt: A numpy array containing the target embedding space. One embedding per row. Number of columns is equal to embedding dimensionality
        :return: The transformation matrix. When multiplying this matrix with src, the resulting embeddings are aligned to the embedding space of tgt.
        """

        return np.ndarray()
