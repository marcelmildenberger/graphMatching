from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class Matcher(ABC):

    @abstractmethod
    def match(self, alice_data: np.ndarray, alice_uids: List[str], eve_data: np.ndarray,
              eve_uids: List[str]) -> Dict[str, str]:
        """
        Creates a Mapping between the two datasets and returns it as a dictionary.
        :param alice_data: A numpy array containing the (aligned) embeddings of Alice
        :param alice_uids: A list of the UIDs of the records contained in alice_data. Order of rows in alice_data must be equal to order in alice_uids.
        :param eve_data: A numpy array containing the (aligned) embeddings of Eve
        :param eve_uids: A list of the UIDs of the records contained in eve_data. Order of rows in alice_data must be equal to order in eve_uids.
        :return: The mapping, a dictionary with UIDs of the smaller dataset as Keys and UIDs of the larger dataset as values. Prefixed with S_ and L_
        """

        # By convention, Alice's data is treated as the larger one if both are of the same size.
        return {"S_UID1": "L_UID1"}
