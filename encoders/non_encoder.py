# Does not encode the data, but rather computes the similarities on the plaintext data

from typing import Sequence, AnyStr, List, Tuple, Any, Union
from tqdm import tqdm
from .encoder import Encoder


def q_gram_dice_sim(q_gram_set1, q_gram_set2):
    """calculate the dice similarity between two given sets of q-grams.
       Dice sim(A,B)= 2 x number of common elements of A and B
                      ----------------------------------------
                      number of elements in A + number of elements in B
       returns a similarity value between 0 and 1.
    """

    num_common_q_gram = len(q_gram_set1 & q_gram_set2)

    q_gram_dice_sim = (2.0 * num_common_q_gram) / \
                      (len(q_gram_set1) + len(q_gram_set2))
    assert 0 <= q_gram_dice_sim and q_gram_dice_sim <= 1.0

    return q_gram_dice_sim


# -----------------------------------------------------------------------------

def q_gram_jacc_sim(q_gram_set1, q_gram_set2):
    """calculate the jaccard similarity between two given sets of q-grams.
       Jaccard sim(A,B) = |A intersection B|
                          ------------------
                              |A Union B|
       returns a value between 0 and 1.
    """

    q_gram_intersection_set = q_gram_set1 & q_gram_set2
    q_gram_union_set = q_gram_set1 | q_gram_set2

    q_gram_jacc_sim = float(len(q_gram_intersection_set) / len(q_gram_union_set))

    assert 0 <= q_gram_jacc_sim and q_gram_jacc_sim <= 1

    return q_gram_jacc_sim



class NonEncoder(Encoder):

    def __init__(self, ngram_size: int, verbose: bool = False):
        """

        """
        self.ngram_size = ngram_size
        self.verbose = verbose

    def encode_and_compare(self, data, uids, metric, sim=True):
        available_metrics = ["jaccard", "dice"]

        pw_metrics = []
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        data = ["".join(d).replace(" ", "").lower() for d in data]
        # Split each string in the data into a list of qgrams to process
        data = [[b[i:i + self.ngram_size] for i in range(len(b) - self.ngram_size + 1)] for b in data]

        for i, q_i in tqdm(enumerate(data), desc="Encoding", total=len(data), disable=not self.verbose):
            for j, q_j in enumerate(data[i + 1:]):
                if metric == "jaccard":
                    val = q_gram_jacc_sim(set(q_i), set(q_j))
                else:
                    val = q_gram_dice_sim(set(q_i), set(q_j))
                if not sim:
                    val = 1 - val
                pw_metrics.append((uids[i], uids[j + i + 1], val))

        return pw_metrics
