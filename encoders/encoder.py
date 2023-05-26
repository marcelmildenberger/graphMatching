# Specifies an interface that must be implemented by encoders in order to be used in the graph matching attack
import numpy as np
from abc import ABC, abstractmethod
from typing import Sequence
class Encoder(ABC):

    @abstractmethod
    def encode_and_compare(self, data: Sequence[Sequence[str]]) -> np.ndarray:
        """
        Encodes the passed data using an encoding algorithm and returns a numpy array containing the
        pairwise similarities of the records.
        :param data: The data to encode as a list of lists: [["Rec1Attr1","Rec1Attr2"],["Rec2Attr1","Rec2Attr2"]]
        :return: A numpy array with the pairwise similarities of the encoded records
        """
        return np.ndarray()
