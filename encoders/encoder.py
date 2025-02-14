# Specifies an interface that must be implemented by encoders in order to be used in the graph matching attack
from abc import ABC, abstractmethod
from typing import Sequence, List, Tuple

import numpy as np


class Encoder(ABC):

    @abstractmethod
    def encode_and_compare_and_append(self, data: Sequence[Sequence[str]], uids: List[int], metric: str,
                           sim: bool, store_encs: bool) -> Tuple[np.ndarray, Sequence[Sequence[str]]]:
        """
        A method that outputs the pairwise similarities of the encoded records as an edgelist.
        :param data: The plaintext data. A list of lists of strings: [["Rec1Attr1", "Rec1Attr2"],["Rec2Attr1", "Rec2Attr2"],...]
        :param uids: A list of numeric UIDs. Ordering of UIDs and data must be identical, i.e. the i-th element of data refers to the record identified by the i-th UID
        :param metric: The similarity metric to be computed on the encoded data.
        :param sim: If true, similarities are returned, else distances.
        :param store_encs: If True, stores the encodings in a dictionary.
        :return: - The edgelist as a numpy array [[UID1, UID2, sim(1,2)], [UID1, UID3, sim(1,3)],...]
                 - An updated dataset (a list of lists of strings)
        """

        # When implementing the store_encs functionality, you should create a dictionary with UIDs as keys and
        # encodings as values. Store it in ./graphMatching/data/encodings

        # For full compatibility, your encoder should be able to handle Dice and Jaccard Similarities
        return np.zeros((5, 3), dtype=np.float32), [["Placeholder1", "Placeholder2"], ["Placeholder3", "Placeholder4"]]
