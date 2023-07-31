# Does not encode the data, but rather computes the similarities on the plaintext data

from typing import Sequence, AnyStr, List, Tuple, Any, Union
from .encoder import Encoder
from strsimpy.sorensen_dice import SorensenDice
from strsimpy.jaccard import Jaccard

class NonEncoder(Encoder):

    def __init__(self, ngram_size: int):
        """

        """
        self.ngram_size = ngram_size

    def encode_and_compare(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str],
                           metric: str, sim: bool = True) -> List[Tuple[int, int, float]]:
        """

        """

        available_metrics = ["dice", "jaccard"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        if metric == "dice":
            comparator = SorensenDice(self.ngram_size)
        else:
            comparator = Jaccard(self.ngram_size)
        pw_metrics = []
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                if sim:
                    val = comparator.similarity("".join(data[i]).replace(" ",""), "".join(data[j]).replace(" ",""))
                else:
                    val = comparator.distance("".join(data[i]).replace(" ",""), "".join(data[j]).replace(" ",""))
                pw_metrics.append((uids[i], uids[j], val))

        return pw_metrics

