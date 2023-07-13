# Does not encode the data, but rather computes the similarities on the plaintext data

from typing import Sequence, AnyStr, List, Tuple, Any, Union
from .encoder import Encoder
from strsimpy.sorensen_dice import SorensenDice

class NonEncoder(Encoder):

    def __init__(self, ngram_size: int):
        """

        """
        self.comparator = SorensenDice(ngram_size)

    def encode_and_compare(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str],
                           metric: str, sim: bool = True) -> List[Tuple[int, int, float]]:
        """

        """
        available_metrics = ["dice"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        pw_metrics = []
        for i in range(len(data)):
            for j in range(i, len(data)):
                if sim:
                    metric = self.comparator.similarity(" ".join(data[i]), " ".join(data[j]))
                else:
                    metric = self.comparator.distance(" ".join(data[i]), " ".join(data[j]))
                pw_metrics.append((uids[i], uids[j], metric))

        return pw_metrics

