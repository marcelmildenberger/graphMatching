from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import numpy as np

class Embedder(ABC):

    @abstractmethod
    def train(self, data: Union[np.ndarray, Union[str, List[Tuple[Union[int, float]]]]]):
        """
        Computes the Node2Vec Embeddings for the graph specified in data but only stores them internally.

        :param data: The graph to encode. Either a string containing the path to an edgelist file, or an edgelist/
         numpy array of the form [(source, target, weight), (source, target, weight), ...]
        :return: Nothing
        """

    @abstractmethod
    def get_vectors(self, ordering: List[str] = None) -> Tuple[np.ndarray, List[Union[int, str]]]:
        """
        Given an ordering (a list of node IDs), returns a numpy array storing their respective embeddings, as well as
        the ordering itself. The ordering of the array (row indices) is equivalent to the odering of the supplied list.
        If no ordering is specified, embeddings of all nodes are returned using the order of the keys of the index dict.
        :param ordering: An ordered list of node ids to retrieve the embeddings for
        :return: The embeddings and the ordering
        """
        return np.ndarray(), ["UID1", "UID2"]