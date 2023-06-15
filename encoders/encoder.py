# Specifies an interface that must be implemented by encoders in order to be used in the graph matching attack
import numpy as np
from abc import ABC, abstractmethod
from typing import Sequence, List, Tuple
class Encoder(ABC):

    @abstractmethod
    def encode_and_compare(self, data: Sequence[Sequence[str]]) -> List[Tuple[int, int, float]]:

        return [(1, 1, 1), (2, 2, 2)]
