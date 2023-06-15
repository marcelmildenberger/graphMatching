# Encodes a given using bloom filters for PPRL

from typing import Sequence, AnyStr, List, Tuple, Any
from encoder import Encoder
import numpy as np
from clkhash import clk
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison

from scipy.spatial.distance import pdist
from utils import calc_condensed_index


class BFEncoder(Encoder):

    def __init__(self, secret: AnyStr, filter_size: int, bits_per_feature: Union[int, Sequence[int]],
                 ngram_size: Union[int, Sequence[int]]):
        self.secret = secret
        self.filter_size = filter_size
        self.bits_per_feature = bits_per_feature
        self.ngram_size = ngram_size

    def __create_schema(self, data: Sequence[Sequence[Union[str, int]]]):
        """
        Creates a linking schema for the CLKhash library based on the parameters specified during creation of the
        Encoder.
        :param data: The data to encode
        :return: Nothing.
        """
        fields = []
        i = 0
        for feature in data[0]:
            # Set StringSpec for string features and IntegerSpec for int features. Note: Right now,
            # only String and Integer features are allowed. Also, the data type at a specific index must be the same
            # across all records.
            if type(feature) == str:
                fields.append(StringSpec(str(i),
                                         FieldHashingProperties(comparator=NgramComparison(
                                             self.ngram_size if type(self.ngram_size) == int else self.ngram_size[i]),
                                             strategy=BitsPerFeatureStrategy(
                                                 self.bits_per_feature if type(self.bits_per_feature) == int else
                                                 self.bits_per_feature[i]
                                             ))))
            else:
                fields.append(IntegerSpec(str(i), FieldHashingProperties(comparator=NgramComparison(2),
                                                                         strategy=BitsPerFeatureStrategy(30))))
            i += 1

        self.schema = Schema(fields, self.filter_size)

    def encode(self, data: Sequence[Sequence[Union[str, int]]]) -> np.ndarray:
        self.__create_schema()
        enc_data = clk.generate_clks(data, self.schema, self.secret)  # Returns a list of bitarrays
        # Convert the bitarrays into lists of bits, then stack them into a numpy array. Cannot stack directly, because
        # numpy would then pack the bits (https://numpy.org/doc/stable/reference/generated/numpy.packbits.html)
        enc_data = np.stack([list(barr) for barr in enc_data])
        return enc_data

    def encode_and_compare(self, data: Sequence[Sequence[str]], metric: str, sim: bool = True) -> List[
        Tuple[int, int, float]]:

        available_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
                             "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis",
                             "matching",
                             "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                             "sqeuclidean", "yule"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        enc = self.encode(data)
        # Calculates pairwise distances between the provided data points
        pw_metrics = pdist(enc, metric=metric)
        # Convert to similarities if specified
        if sim:
            pw_metrics = [1 - p for p in pw_metric]

        # Convert to "long" format
        pw_metrics_long = []
        for i in range(len(enc)):
            for j in range(i + 1, len(enc)):
                pw_metrics_long.append((i, j, pw_metrics[calc_condensed_index(i, j, len(enc))]))

        return pw_metrics_long
