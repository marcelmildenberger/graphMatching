# Specifies an interface that must be implemented by encoders in order to be used in the graph matching attack
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np


def validate_metric(metric: str, allowed_metrics: Sequence[str], *, label: str = "similarity metric") -> str:
    if metric not in allowed_metrics:
        raise ValueError(f"Invalid {label}. Must be one of {list(allowed_metrics)}")
    return metric


class Encoder(ABC):

    @abstractmethod
    def encode_and_compare(self, data: Sequence[Sequence[str]], uids: List[int], metric: str,
                           sim: bool, store_encs: bool) -> np.ndarray:
        """
        Output the pairwise similarities of the encoded records as an edge list.
        :param data: The plaintext data. A list of lists of strings:
            [["Rec1Attr1", "Rec1Attr2"], ["Rec2Attr1", "Rec2Attr2"], ...]
        :param uids: Numeric UIDs in the same order as `data`, i.e. the i-th element
            of `data` refers to the record identified by the i-th UID.
        :param metric: The similarity metric to be computed on the encoded data.
        :param sim: If true, similarities are returned; otherwise distances.
        :param store_encs: If True, stores the encodings in a dictionary.
        :return: The edgelist as a numpy array [[UID1, UID2, sim(1,2)], [UID1, UID3, sim(1,3)],...]
        """

        # When implementing the store_encs functionality, you should create a dictionary with UIDs as keys and
        # encodings as values. Store it in ./graphMatching/data/encodings

        # For full compatibility, your encoder should be able to handle Dice and Jaccard Similarities
        return np.zeros((5, 3), dtype=np.float32)
