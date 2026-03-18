from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np

PAIRWISE_DISTANCE_METRICS: Tuple[str, ...] = (
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "l1",
    "l2",
    "manhattan",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
)


def validate_distance_metric(metric: str) -> str:
    if metric not in PAIRWISE_DISTANCE_METRICS:
        raise ValueError(f"Invalid similarity metric. Must be one of {list(PAIRWISE_DISTANCE_METRICS)}")
    return metric


class Matcher(ABC):

    @abstractmethod
    def match(self, alice_data: np.ndarray, alice_uids: List[str], eve_data: np.ndarray,
              eve_uids: List[str]) -> Dict[str, str]:
        """
        Create a mapping between the two datasets and return it as a dictionary.
        :param alice_data: A numpy array containing the aligned embeddings of Alice.
        :param alice_uids: UIDs of the records in `alice_data`, in row order.
        :param eve_data: A numpy array containing the aligned embeddings of Eve.
        :param eve_uids: UIDs of the records in `eve_data`, in row order.
        :return: A mapping with UIDs of the smaller dataset as keys and UIDs of the
            larger dataset as values, prefixed with `S_` and `L_`.
        """

        # By convention, Alice's data is treated as the larger one if both are of the same size.
        return {"S_UID1": "L_UID1"}
